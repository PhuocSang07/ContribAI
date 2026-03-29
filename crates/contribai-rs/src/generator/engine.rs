//! LLM-powered contribution generator.
//!
//! Port from Python `generator/engine.py`.
//! Takes findings from analysis and generates actual code changes,
//! tests, and commit messages that follow the target repo's conventions.

use chrono::Utc;
use regex::Regex;
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::core::config::ContributionConfig;
use crate::core::error::Result;
use crate::core::models::{
    Contribution, ContributionType, FileChange, Finding, RepoContext,
};
use crate::github::guidelines::{adapt_pr_title, extract_scope_from_path, RepoGuidelines};
use crate::llm::provider::LlmProvider;

// ── Generator struct ──────────────────────────────────────────────────────────

/// Generate code contributions from analysis findings.
pub struct ContributionGenerator<'a> {
    llm: &'a dyn LlmProvider,
    config: &'a ContributionConfig,
    /// Enable LLM self-review gate after generation (default: true).
    self_review_enabled: bool,
}

impl<'a> ContributionGenerator<'a> {
    pub fn new(llm: &'a dyn LlmProvider, config: &'a ContributionConfig) -> Self {
        Self {
            llm,
            config,
            self_review_enabled: true,
        }
    }

    /// Disable self-review (useful for batch pipelines where latency matters).
    pub fn without_self_review(mut self) -> Self {
        self.self_review_enabled = false;
        self
    }

    /// Generate a contribution for a single finding.
    ///
    /// Pipeline:
    /// 1. Build context-aware prompt
    /// 2. Get LLM to generate the fix
    /// 3. Parse structured output into FileChanges (with search/replace)
    /// 4. Generate commit message
    /// 5. Optional self-review LLM gate
    pub async fn generate(
        &self,
        finding: &Finding,
        context: &RepoContext,
    ) -> Result<Option<Contribution>> {
        self.generate_with_guidelines(finding, context, None).await
    }

    /// Generate a contribution, optionally adapting PR title to repo guidelines.
    pub async fn generate_with_guidelines(
        &self,
        finding: &Finding,
        context: &RepoContext,
        guidelines: Option<&RepoGuidelines>,
    ) -> Result<Option<Contribution>> {
        // 1. Build prompts
        let system = self.build_system_prompt(context);
        let prompt = self.build_generation_prompt(finding, context);

        // 2. Generate with retry (max 1 retry = 2 attempts)
        let mut changes: Option<Vec<FileChange>> = None;
        let mut last_error = String::new();

        for attempt in 0..2 {
            let actual_prompt = if attempt > 0 {
                format!(
                    "{}\n\n## IMPORTANT: Your previous attempt failed.\n\
                     Error: {}\n\
                     Please fix the issue and return ONLY valid JSON \
                     with no markdown fences or extra text.",
                    prompt, last_error
                )
            } else {
                prompt.clone()
            };

            let response = self
                .llm
                .complete(&actual_prompt, Some(&system), Some(0.2), None)
                .await?;

            // 3. Parse changes (search/replace or full-content format)
            match self.parse_changes(&response, context) {
                Some(c) if !c.is_empty() => {
                    if self.validate_changes(&c) {
                        changes = Some(c);
                        break;
                    } else {
                        last_error = "Generated code failed syntax validation \
                                     (unbalanced brackets or empty edits)"
                            .into();
                    }
                }
                _ => {
                    last_error = "No valid changes could be parsed from JSON output".into();
                }
            }
        }

        let changes = match changes {
            Some(c) => c,
            None => {
                warn!(title = %finding.title, "No valid changes after retries");
                return Ok(None);
            }
        };

        // 4. Generate commit message
        let commit_msg = self.generate_commit_message(finding, &changes);

        // 5. Generate branch name
        let branch_name = Self::generate_branch_name(finding);

        // 6. Generate PR title (adapted to guidelines if available)
        let pr_title = Self::generate_pr_title_with_guidelines(finding, guidelines);

        let contribution = Contribution {
            finding: finding.clone(),
            contribution_type: finding.finding_type.clone(),
            title: pr_title,
            description: finding.description.clone(),
            changes,
            commit_message: commit_msg,
            tests_added: vec![],
            branch_name,
            generated_at: Utc::now(),
        };

        // 7. Optional self-review LLM gate
        if self.self_review_enabled {
            let approved = self.self_review(&contribution, context).await;
            if !approved {
                warn!(title = %finding.title, "Self-review rejected contribution");
                return Ok(None);
            }
        }

        info!(
            title = %contribution.title,
            files = contribution.total_files_changed(),
            "Generated contribution"
        );

        Ok(Some(contribution))
    }

    // ── Prompt builders ───────────────────────────────────────────────────────

    /// Build system prompt with repo context and style guidance.
    fn build_system_prompt(&self, context: &RepoContext) -> String {
        let mut prompt = String::from(
            "You are a senior open-source contributor who writes production-ready \
             code. You understand that PRs are judged by maintainers who value \
             minimal, focused, and convention-matching changes.\n\n\
             RULES FOR GENERATING CHANGES:\n\
             1. Match existing code style EXACTLY (indentation, naming, patterns)\n\
             2. Make the SMALLEST change that correctly fixes the issue\n\
             3. Include proper error handling consistent with the codebase\n\
             4. Do NOT break existing functionality\n\
             5. Do NOT add unnecessary dependencies or imports\n\
             6. Do NOT refactor adjacent code — fix only the reported issue\n\
             7. Do NOT add comments explaining what the code does\n\
             8. Do NOT modify files unrelated to the finding\n\n\
             OUTPUT FORMAT RULES (CRITICAL):\n\
             - Return ONLY raw JSON — no markdown fences, no ```json blocks\n\
             - No explanatory text before or after the JSON\n\
             - The response must be valid, parseable JSON and nothing else\n\n\
             ACCEPTANCE CRITERIA:\n\
             - Would a busy maintainer merge this in under 30 seconds?\n\
             - Is the change obviously correct with no side effects?\n",
        );

        if let Some(style) = &context.coding_style {
            prompt.push_str(&format!(
                "\nCODEBASE STYLE:\n{}\n\
                 You MUST match these conventions exactly.\n",
                style
            ));
        }

        prompt.push_str(&format!(
            "\nREPOSITORY: {}\nLanguage: {}\n",
            context.repo.full_name,
            context.repo.language.as_deref().unwrap_or("unknown")
        ));

        prompt
    }

    /// Build the generation prompt based on finding.
    ///
    /// Uses search/replace format for existing files (matching Python engine).
    fn build_generation_prompt(&self, finding: &Finding, context: &RepoContext) -> String {
        let current_content = context
            .relevant_files
            .get(&finding.file_path)
            .map(|s| s.as_str())
            .unwrap_or("");

        let suggestion_line = finding
            .suggestion
            .as_deref()
            .map(|s| format!("- **Suggestion**: {}\n", s))
            .unwrap_or_default();

        let mut prompt = format!(
            "## Task\nFix this issue.\n\n\
             ## Finding\n\
             - **Title**: {}\n\
             - **Severity**: {}\n\
             - **File**: {}\n\
             - **Description**: {}\n\
             {}",
            finding.title,
            finding.severity,
            finding.file_path,
            finding.description,
            suggestion_line
        );

        // Cross-file: find other files with the same issue pattern
        let cross_files = self.find_cross_file_instances(finding, context);
        if !cross_files.is_empty() {
            prompt.push_str(&format!(
                "\n## IMPORTANT: Same issue in {} OTHER file(s)\n\
                 Fix ALL instances across ALL files in a single contribution.\n\n",
                cross_files.len()
            ));
            for (fpath, fcontent) in &cross_files {
                let snippet = if fcontent.len() > 3000 {
                    &fcontent[..3000]
                } else {
                    fcontent.as_str()
                };
                prompt.push_str(&format!("### {}\n```\n{}\n```\n\n", fpath, snippet));
            }
        }

        prompt.push_str("\n## Output Format\nReturn your changes as a JSON object.\n\n");

        if !current_content.is_empty() {
            let snippet = if current_content.len() > 6000 {
                &current_content[..6000]
            } else {
                current_content
            };
            prompt.push_str(&format!(
                "## Current File Content ({})\n```\n{}\n```\n\n",
                finding.file_path, snippet
            ));
            prompt.push_str(
                "Since this is an EXISTING file, use SEARCH/REPLACE blocks \
                 to make targeted edits. DO NOT rewrite the entire file.\n\n\
                 ```json\n\
                 {{\n  \"changes\": [\n    {{\n\
                       \"path\": \"path/to/file\",\n\
                       \"is_new_file\": false,\n\
                       \"edits\": [\n        {{\n\
                           \"search\": \"exact text to find in the file\",\n\
                           \"replace\": \"replacement text\"\n\
                       }}\n      ]\n    }}\n  ]\n}}\n\
                 ```\n\n\
                 RULES for search/replace:\n\
                 - `search` must be an EXACT substring from the current file\n\
                 - `replace` is what replaces it (can be longer/shorter)\n\
                 - To DELETE content, set `replace` to empty string\n\
                 - Keep each edit small and focused\n",
            );
        } else {
            prompt.push_str(
                "Since this is a NEW file, provide the full content:\n\n\
                 ```json\n\
                 {{\n  \"changes\": [\n    {{\n\
                       \"path\": \"path/to/file\",\n\
                       \"content\": \"full content of the new file\",\n\
                       \"is_new_file\": true\n\
                   }}\n  ]\n}}\n\
                 ```\n",
            );
        }

        prompt
    }

    // ── JSON extraction ───────────────────────────────────────────────────────

    /// Robustly extract JSON from LLM response.
    ///
    /// Three strategies (matching Python `_extract_json`):
    /// 1. Extract from ` ```json ` fences
    /// 2. Plain ` ``` ` fences containing a JSON object
    /// 3. Bracket-counting fallback from first `{"changes"`, `{`, or `[`
    pub fn extract_json(response: &str) -> Option<String> {
        // Strategy 1: ```json ... ``` fenced blocks
        if let Ok(re) = Regex::new(r"(?s)```json\s*\n(.*?)\n\s*```") {
            if let Some(cap) = re.captures(response) {
                if let Some(m) = cap.get(1) {
                    return Some(m.as_str().trim().to_string());
                }
            }
        }

        // Strategy 2: plain ``` ... ``` fence containing a JSON object
        if let Ok(re) = Regex::new(r"(?s)```\s*\n(\{.*?\})\s*\n\s*```") {
            if let Some(cap) = re.captures(response) {
                if let Some(m) = cap.get(1) {
                    return Some(m.as_str().trim().to_string());
                }
            }
        }

        // Strategy 3: bracket-counting — prefer `{"changes"` anchor, then first `[` or `{`
        // whichever comes first in the text (so bare arrays are not missed).
        let start = response
            .find(r#"{"changes""#)
            .or_else(|| {
                let brace = response.find('{');
                let bracket = response.find('[');
                match (brace, bracket) {
                    (Some(b), Some(k)) => Some(b.min(k)),
                    (Some(b), None) => Some(b),
                    (None, Some(k)) => Some(k),
                    _ => None,
                }
            });

        let start = start?;

        let open_ch = response.as_bytes().get(start).copied()? as char;
        let close_ch = if open_ch == '{' { '}' } else { ']' };

        let mut depth: i32 = 0;
        let mut in_string = false;
        let mut prev_ch = '\0';

        for (i, ch) in response[start..].char_indices() {
            if ch == '"' && prev_ch != '\\' {
                in_string = !in_string;
            }
            if in_string {
                prev_ch = ch;
                continue;
            }
            if ch == open_ch {
                depth += 1;
            } else if ch == close_ch {
                depth -= 1;
                if depth == 0 {
                    let end = start + i + ch.len_utf8();
                    return Some(response[start..end].to_string());
                }
            }
            prev_ch = ch;
        }

        None
    }

    // ── Change parsing ────────────────────────────────────────────────────────

    /// Parse LLM response into FileChange objects.
    ///
    /// Supports two formats (matching Python engine):
    /// 1. `{"changes": [{"path":..., "edits": [{"search":..., "replace":...}]}]}`
    ///    — search/replace applied to original file content from `context`
    /// 2. `{"changes": [{"path":..., "content":..., "is_new_file": true}]}`
    ///    — full content for new files
    ///
    /// Falls back to bare JSON array `[{"path":..., "new_content":...}]` for
    /// backward compatibility.
    fn parse_changes(&self, response: &str, context: &RepoContext) -> Option<Vec<FileChange>> {
        let json_text = Self::extract_json(response)?;

        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&json_text) {
            // Try canonical `{"changes": [...]}` wrapper format first
            if let Some(raw_changes) = data.get("changes").and_then(|v| v.as_array()) {
                let changes = self.apply_changes_from_json(raw_changes, context);
                if !changes.is_empty() {
                    return Some(changes);
                }
            }

            // Bare array fallback: [{path, new_content, is_new_file}]
            if let Some(items) = data.as_array() {
                let changes = Self::parse_bare_array(items);
                if !changes.is_empty() {
                    return Some(changes);
                }
            }
        }

        None
    }

    /// Apply search/replace edits or full-content changes from JSON items.
    fn apply_changes_from_json(
        &self,
        items: &[serde_json::Value],
        context: &RepoContext,
    ) -> Vec<FileChange> {
        let mut changes: Vec<FileChange> = Vec::new();

        for item in items {
            let path = match item.get("path").and_then(|v| v.as_str()) {
                Some(p) => p.to_string(),
                None => continue,
            };

            let is_new = item
                .get("is_new_file")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if let Some(edits) = item.get("edits").and_then(|v| v.as_array()) {
                // Search/replace mode — requires original file content
                let original = match context.relevant_files.get(&path) {
                    Some(c) => c.clone(),
                    None => {
                        warn!(path = %path, "No original content for search/replace edits");
                        continue;
                    }
                };

                let mut new_content = original.clone();
                let edits_total = edits.len();
                let mut edits_applied: usize = 0;

                for edit in edits {
                    let search = edit
                        .get("search")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let replace = edit
                        .get("replace")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    if search.is_empty() {
                        continue;
                    }

                    if let Some(updated) =
                        Self::apply_single_edit(&new_content, &search, &replace, &path)
                    {
                        new_content = updated;
                        edits_applied += 1;
                    } else {
                        warn!(
                            path = %path,
                            search_len = search.len(),
                            search_preview = %&search[..search.len().min(80)].replace('\n', "\\n"),
                            "Search text not found (tried exact + 3 fuzzy strategies)"
                        );
                    }
                }

                info!(
                    path = %path,
                    applied = edits_applied,
                    total = edits_total,
                    "Edits applied"
                );

                if edits_applied == 0 {
                    warn!(path = %path, "No edits applied, skipping file");
                    continue;
                }

                changes.push(FileChange {
                    path,
                    original_content: Some(original),
                    new_content,
                    is_new_file: false,
                    is_deleted: false,
                });
            } else if let Some(content) = item.get("content").and_then(|v| v.as_str()) {
                // Full-content mode (new files or fallback)
                changes.push(FileChange {
                    path,
                    original_content: None,
                    new_content: content.to_string(),
                    is_new_file: true,
                    is_deleted: false,
                });
            }
        }

        // Enforce max files limit
        let max = self.config.max_changes_per_pr;
        if changes.len() > max {
            warn!(
                actual = changes.len(),
                limit = max,
                "Too many files changed, truncating"
            );
            changes.truncate(max);
        }

        changes
    }

    /// Parse legacy bare-array format `[{"path":..., "new_content":..., "is_new_file":...}]`.
    fn parse_bare_array(items: &[serde_json::Value]) -> Vec<FileChange> {
        items
            .iter()
            .filter_map(|item| {
                let path = item.get("path")?.as_str()?.to_string();
                let new_content = item.get("new_content")?.as_str()?.to_string();
                let is_new_file = item
                    .get("is_new_file")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                Some(FileChange {
                    path,
                    original_content: None,
                    new_content,
                    is_new_file,
                    is_deleted: false,
                })
            })
            .collect()
    }

    // ── Fuzzy matching ────────────────────────────────────────────────────────

    /// Apply a single search/replace edit using 4 strategies with graceful fallback.
    ///
    /// Strategy order (matches Python `_parse_changes` edit loop):
    /// 1. Exact substring match
    /// 2. Normalized trailing-whitespace match (per line)
    /// 3. Stripped leading/trailing whitespace match
    /// 4. Token-based similarity (word overlap ratio >= 0.8)
    fn apply_single_edit(
        content: &str,
        search: &str,
        replace: &str,
        path: &str,
    ) -> Option<String> {
        // Strategy 1: exact
        if content.contains(search) {
            return Some(content.replacen(search, replace, 1));
        }

        // Strategy 2: normalize trailing whitespace per line
        {
            let norm_search: String = search
                .split('\n')
                .map(|l| l.trim_end())
                .collect::<Vec<_>>()
                .join("\n");
            let norm_content: String = content
                .split('\n')
                .map(|l| l.trim_end())
                .collect::<Vec<_>>()
                .join("\n");

            if let Some(idx) = norm_content.find(&norm_search) {
                let start_line = norm_content[..idx].matches('\n').count();
                let end_line = start_line + norm_search.matches('\n').count();
                let mut lines: Vec<&str> = content.split('\n').collect();
                let replace_lines: Vec<&str> = replace.split('\n').collect();
                lines.splice(start_line..=end_line, replace_lines);
                debug!(path = %path, "Fuzzy match (whitespace normalized)");
                return Some(lines.join("\n"));
            }
        }

        // Strategy 3: strip all leading/trailing whitespace
        {
            let stripped = search.trim();
            if stripped.len() > 20 && content.contains(stripped) {
                debug!(path = %path, "Fuzzy match (stripped)");
                return Some(content.replacen(stripped, replace.trim(), 1));
            }
        }

        // Strategy 4: token-based similarity (word overlap Dice coefficient >= 0.8)
        if search.len() > 20 {
            if let Some(result) = Self::fuzzy_replace(content, search, replace) {
                debug!(path = %path, "Fuzzy match (token similarity)");
                return Some(result);
            }
        }

        None
    }

    /// Find the best-matching block in `content` using word-overlap similarity.
    ///
    /// Slides a window the same number of lines as `search` over `content`.
    /// Uses Dice coefficient on word sets: `2 * |intersection| / (|A| + |B|)`.
    /// Returns modified content if best ratio >= 0.8, otherwise `None`.
    pub fn fuzzy_replace(content: &str, search: &str, replace: &str) -> Option<String> {
        let search_lines: Vec<&str> = search.lines().collect();
        let content_lines: Vec<&str> = content.lines().collect();
        let search_len = search_lines.len();

        if search_len == 0 || search_len > content_lines.len() {
            return None;
        }

        let search_words: Vec<&str> = search.split_whitespace().collect();

        let mut best_ratio = 0.0_f64;
        let mut best_start: Option<usize> = None;

        for i in 0..=(content_lines.len() - search_len) {
            let window = content_lines[i..i + search_len].join("\n");
            let window_words: Vec<&str> = window.split_whitespace().collect();

            let ratio = word_overlap_ratio(&search_words, &window_words);
            if ratio > best_ratio {
                best_ratio = ratio;
                best_start = Some(i);
            }
        }

        if best_ratio >= 0.8 {
            if let Some(start) = best_start {
                let replace_lines: Vec<&str> = replace.lines().collect();
                let mut result = content_lines[..start].to_vec();
                result.extend_from_slice(&replace_lines);
                result.extend_from_slice(&content_lines[start + search_len..]);
                return Some(result.join("\n"));
            }
        }

        None
    }

    // ── Validation ────────────────────────────────────────────────────────────

    /// Validate generated changes for basic sanity.
    ///
    /// Checks:
    /// - Non-empty content for new files
    /// - No-op detection (original == new)
    /// - Balanced brackets (string/comment-aware)
    fn validate_changes(&self, changes: &[FileChange]) -> bool {
        if changes.is_empty() {
            return false;
        }

        for change in changes {
            let content = &change.new_content;

            // New file must have non-trivial content
            if change.is_new_file && content.trim().len() < 10 {
                debug!(
                    path = %change.path,
                    len = content.trim().len(),
                    "Validation: new file content too short"
                );
                return false;
            }

            // Detect no-op edits
            if let Some(orig) = &change.original_content {
                if content == orig {
                    debug!(path = %change.path, "Validation: new_content identical to original (no-op)");
                    return false;
                }
            }

            // Balanced bracket check (string/comment-aware)
            if !content.is_empty() {
                let unbalanced = Self::count_unbalanced_brackets(content);
                if unbalanced > 5 {
                    debug!(
                        path = %change.path,
                        unbalanced = unbalanced,
                        "Validation: too many unbalanced brackets"
                    );
                    return false;
                }
            }
        }

        true
    }

    /// Count unbalanced brackets, ignoring those inside strings and comments.
    ///
    /// Handles:
    /// - Single-line comments: `#` (Python) and `//` (C-like)
    /// - Block comments: `/* ... */`
    /// - String literals delimited by `"` or `'` (with backslash-escape tracking)
    pub fn count_unbalanced_brackets(code: &str) -> usize {
        let open_to_close: HashMap<char, char> =
            [('(', ')'), ('[', ']'), ('{', '}')].into_iter().collect();
        let closers: std::collections::HashSet<char> = [')', ']', '}'].into_iter().collect();

        let mut stack: Vec<char> = Vec::new();
        let mut in_string: Option<char> = None; // current quote character
        let mut in_line_comment = false;
        let mut in_block_comment = false;
        let chars: Vec<char> = code.chars().collect();
        let n = chars.len();
        let mut i = 0;

        while i < n {
            let ch = chars[i];
            let next = chars.get(i + 1).copied();

            // Newline resets line comments
            if ch == '\n' {
                in_line_comment = false;
                i += 1;
                continue;
            }

            // Skip chars inside line comments
            if in_line_comment {
                i += 1;
                continue;
            }

            // Handle block comment end
            if in_block_comment {
                if ch == '*' && next == Some('/') {
                    in_block_comment = false;
                    i += 2; // consume "*/"
                    continue;
                }
                i += 1;
                continue;
            }

            // Handle block comment start (outside strings)
            if in_string.is_none() && ch == '/' && next == Some('*') {
                in_block_comment = true;
                i += 2;
                continue;
            }

            // Handle line comment start: `#` or `//`
            if in_string.is_none() {
                if ch == '#' {
                    in_line_comment = true;
                    i += 1;
                    continue;
                }
                if ch == '/' && next == Some('/') {
                    in_line_comment = true;
                    i += 2;
                    continue;
                }
            }

            // Handle string boundaries (skip escaped quotes)
            if (ch == '"' || ch == '\'') && (i == 0 || chars[i - 1] != '\\') {
                match in_string {
                    None => in_string = Some(ch),
                    Some(q) if q == ch => in_string = None,
                    _ => {} // inside a different-quote string
                }
                i += 1;
                continue;
            }

            // Skip chars inside strings
            if in_string.is_some() {
                i += 1;
                continue;
            }

            // Count brackets
            if let Some(&close) = open_to_close.get(&ch) {
                stack.push(close);
            } else if closers.contains(&ch) {
                if stack.last() == Some(&ch) {
                    stack.pop();
                }
            }

            i += 1;
        }

        stack.len()
    }

    // ── Cross-file detection ──────────────────────────────────────────────────

    /// Find other files in the repo with the same issue pattern.
    ///
    /// Searches `context.relevant_files` for code patterns extracted from the
    /// finding description/suggestion. Returns `{path: content}` for files
    /// with at least 2 keyword matches (capped at 3 extra files).
    pub fn find_cross_file_instances(
        &self,
        finding: &Finding,
        context: &RepoContext,
    ) -> HashMap<String, String> {
        if finding.file_path.is_empty() || context.relevant_files.is_empty() {
            return HashMap::new();
        }

        let keywords = Self::extract_search_patterns(finding);
        if keywords.is_empty() {
            return HashMap::new();
        }

        let mut other_files: HashMap<String, String> = HashMap::new();

        for (fpath, content) in &context.relevant_files {
            if fpath == &finding.file_path {
                continue;
            }
            let content_lower = content.to_lowercase();
            let matches = keywords
                .iter()
                .filter(|kw| content_lower.contains(kw.to_lowercase().as_str()))
                .count();

            if matches >= 2 {
                other_files.insert(fpath.clone(), content.clone());
                if other_files.len() >= 3 {
                    break;
                }
            }
        }

        if !other_files.is_empty() {
            info!(
                count = other_files.len(),
                files = ?other_files.keys().collect::<Vec<_>>(),
                "Found same pattern in other files"
            );
        }

        other_files
    }

    /// Extract code patterns from finding description and suggestion.
    ///
    /// Looks for backtick-quoted snippets (`foo`) and dotted identifiers (foo.bar()).
    fn extract_search_patterns(finding: &Finding) -> Vec<String> {
        let text = format!(
            "{} {}",
            finding.description,
            finding.suggestion.as_deref().unwrap_or("")
        );

        let mut patterns: Vec<String> = Vec::new();

        // Backtick-quoted snippets
        if let Ok(re) = Regex::new(r"`([^`]+)`") {
            for cap in re.captures_iter(&text) {
                if let Some(m) = cap.get(1) {
                    let s = m.as_str().to_string();
                    if s.len() > 3 {
                        patterns.push(s);
                    }
                }
            }
        }

        // Dotted identifiers (e.g., `foo.bar()`, `obj.method!`)
        if let Ok(re) = Regex::new(r"(\w+\.\w+[!?]?(?:\(\))?)") {
            for cap in re.captures_iter(&text) {
                if let Some(m) = cap.get(1) {
                    let s = m.as_str().to_string();
                    if s.len() > 5 {
                        patterns.push(s);
                    }
                }
            }
        }

        patterns.truncate(10);
        patterns
    }

    // ── Self-review LLM gate ──────────────────────────────────────────────────

    /// Have the LLM review the generated contribution and approve or reject it.
    ///
    /// Builds a unified diff for modified files and asks the LLM whether the
    /// change is correct. Defaults to `true` (approved) on LLM failures to
    /// avoid blocking contributions on transient errors.
    async fn self_review(&self, contribution: &Contribution, context: &RepoContext) -> bool {
        let changes_summary: String = contribution
            .changes
            .iter()
            .map(|c| {
                format!(
                    "- {} ({})\n",
                    c.path,
                    if c.is_new_file { "new" } else { "modified" }
                )
            })
            .collect();

        let mut prompt = format!(
            "Review the following code contribution for quality:\n\n\
             **Title**: {}\n\
             **Type**: {:?}\n\
             **Finding**: {}\n\
             **Changes**:\n{}\n\n\
             For each changed file:\n",
            contribution.title,
            contribution.contribution_type,
            contribution.finding.description,
            changes_summary
        );

        for change in contribution.changes.iter().take(5) {
            let original = context.relevant_files.get(&change.path);
            if let (Some(orig), false) = (original, change.is_new_file) {
                let diff = unified_diff(orig, &change.new_content, &change.path);
                let diff_snippet = if diff.len() > 4000 {
                    diff[..4000].to_string()
                } else {
                    diff
                };
                prompt.push_str(&format!(
                    "\n### {} (diff)\n```diff\n{}\n```\n",
                    change.path, diff_snippet
                ));
            } else {
                let snippet = if change.new_content.len() > 4000 {
                    &change.new_content[..4000]
                } else {
                    &change.new_content
                };
                prompt.push_str(&format!(
                    "\n### {}\n```\n{}\n```\n",
                    change.path, snippet
                ));
            }
        }

        prompt.push_str(
            "\nAnswer these questions:\n\
             1. Does the change address the described issue?\n\
             2. Does it introduce any obvious new bugs or security vulnerabilities?\n\
             3. Is the change reasonable and follows existing code style?\n\n\
             IMPORTANT: Be lenient. APPROVE if the change is a net improvement, \
             even if minor improvements could be made. Only REJECT if the change \
             is clearly wrong, introduces a bug, or is completely unrelated to the issue.\n\n\
             Reply with APPROVE or REJECT followed by brief reasoning.",
        );

        match self.llm.complete(&prompt, None, Some(0.1), None).await {
            Ok(response) => {
                let approved = response.to_uppercase().contains("APPROVE");
                if !approved {
                    info!(
                        preview = %&response[..response.len().min(200)],
                        "Self-review rejected"
                    );
                }
                approved
            }
            Err(e) => {
                warn!(error = %e, "Self-review LLM call failed, approving by default");
                true
            }
        }
    }

    // ── Commit / branch / PR title ────────────────────────────────────────────

    /// Generate a conventional commit message.
    fn generate_commit_message(&self, finding: &Finding, changes: &[FileChange]) -> String {
        let prefix = match finding.finding_type {
            ContributionType::SecurityFix => "fix(security)",
            ContributionType::CodeQuality => "refactor",
            ContributionType::DocsImprove => "docs",
            ContributionType::PerformanceOpt => "perf",
            ContributionType::FeatureAdd => "feat",
            ContributionType::Refactor => "refactor",
            ContributionType::UiUxFix => "fix(ui)",
        };

        // Extract scope from first changed file path (matching Python logic)
        let scope = changes.first().and_then(|c| {
            let parts: Vec<&str> = c.path.split('/').collect();
            if parts.len() >= 2
                && matches!(parts[0], "src" | "packages" | "apps" | "libs")
            {
                Some(parts[1].to_string())
            } else {
                None
            }
        });

        let title = finding.title.to_lowercase();
        let title = if title.len() > 50 { &title[..50] } else { &title };
        let files: String = changes
            .iter()
            .take(3)
            .map(|c| c.path.split('/').next_back().unwrap_or(&c.path))
            .collect::<Vec<_>>()
            .join(", ");

        if let Some(s) = scope {
            format!(
                "{}({}): {}\n\n{}\n\nAffected files: {}",
                prefix, s, title, finding.description, files
            )
        } else {
            format!(
                "{}: {}\n\n{}\n\nAffected files: {}",
                prefix, title, finding.description, files
            )
        }
    }

    /// Generate a natural-looking branch name.
    pub fn generate_branch_name(finding: &Finding) -> String {
        let prefix = match finding.finding_type {
            ContributionType::SecurityFix => "fix/security",
            ContributionType::CodeQuality => "improve/quality",
            ContributionType::DocsImprove => "docs",
            ContributionType::PerformanceOpt => "perf",
            ContributionType::FeatureAdd => "feat",
            ContributionType::Refactor => "refactor",
            ContributionType::UiUxFix => "fix/ui",
        };

        let re = Regex::new(r"[^a-z0-9]+").unwrap();
        let lower = finding.title.to_lowercase();
        let slug = re.replace_all(&lower, "-");
        let slug = slug.trim_matches('-');
        let slug = if slug.len() > 40 { &slug[..40] } else { slug };

        format!("contribai/{}/{}", prefix, slug)
    }

    /// Generate a PR title using the default label format.
    pub fn generate_pr_title(finding: &Finding) -> String {
        Self::generate_pr_title_with_guidelines(finding, None)
    }

    /// Generate a PR title adapted to repo guidelines if available.
    ///
    /// If `guidelines` is `Some` and `has_guidelines()` returns true, delegates to
    /// `adapt_pr_title` + `extract_scope_from_path` from `guidelines.rs`.
    /// Otherwise falls back to the default label-based format.
    pub fn generate_pr_title_with_guidelines(
        finding: &Finding,
        guidelines: Option<&RepoGuidelines>,
    ) -> String {
        if let Some(g) = guidelines {
            if g.has_guidelines() {
                let scope = extract_scope_from_path(&finding.file_path, g);
                let type_str = match finding.finding_type {
                    ContributionType::SecurityFix => "security_fix",
                    ContributionType::CodeQuality => "code_quality",
                    ContributionType::DocsImprove => "docs_improve",
                    ContributionType::UiUxFix => "ui_ux_fix",
                    ContributionType::PerformanceOpt => "performance_opt",
                    ContributionType::FeatureAdd => "feature_add",
                    ContributionType::Refactor => "refactor",
                };
                return adapt_pr_title(&finding.title, type_str, g, &scope);
            }
        }

        // Default: label-based format
        let label = match finding.finding_type {
            ContributionType::SecurityFix => "Security",
            ContributionType::CodeQuality => "Quality",
            ContributionType::DocsImprove => "Docs",
            ContributionType::UiUxFix => "UI/UX",
            ContributionType::PerformanceOpt => "Performance",
            ContributionType::FeatureAdd => "Feature",
            ContributionType::Refactor => "Refactor",
        };
        format!("{}: {}", label, finding.title)
    }
}

// ── Free-standing helpers ─────────────────────────────────────────────────────

/// Compute Dice coefficient (word overlap ratio) between two word slices.
///
/// Formula: `2 * |intersection| / (|a| + |b|)` where intersection is the
/// multiset intersection (accounts for repeated words).
fn word_overlap_ratio(a: &[&str], b: &[&str]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    // Count word frequencies in b
    let mut b_counts: HashMap<&str, usize> = HashMap::new();
    for w in b {
        *b_counts.entry(w).or_insert(0) += 1;
    }

    // Count intersection (limited by min frequency in each side)
    let mut a_counts: HashMap<&str, usize> = HashMap::new();
    let mut intersection: usize = 0;
    for w in a {
        let a_c = a_counts.entry(w).or_insert(0);
        *a_c += 1;
        let b_c = b_counts.get(w).copied().unwrap_or(0);
        if *a_c <= b_c {
            intersection += 1;
        }
    }

    2.0 * intersection as f64 / (a.len() + b.len()) as f64
}

/// Build a simple diff string between two text blobs for LLM self-review.
///
/// Emits removed lines (prefixed `-`) and added lines (prefixed `+`).
/// This is sufficient for the LLM to understand the nature of the change.
fn unified_diff(original: &str, new_content: &str, path: &str) -> String {
    let orig_lines: Vec<&str> = original.lines().collect();
    let new_lines: Vec<&str> = new_content.lines().collect();

    let mut output = format!("--- a/{}\n+++ b/{}\n", path, path);

    let orig_set: std::collections::HashSet<&str> = orig_lines.iter().copied().collect();
    let new_set: std::collections::HashSet<&str> = new_lines.iter().copied().collect();

    for line in &orig_lines {
        if !new_set.contains(*line) {
            output.push_str(&format!("-{}\n", line));
        }
    }
    for line in &new_lines {
        if !orig_set.contains(*line) {
            output.push_str(&format!("+{}\n", line));
        }
    }

    output
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::models::{ContributionType, Severity};

    fn test_finding() -> Finding {
        Finding {
            id: "test".into(),
            finding_type: ContributionType::SecurityFix,
            severity: Severity::High,
            title: "SQL injection in user query".into(),
            description: "User input not sanitized".into(),
            file_path: "src/db/queries.py".into(),
            line_start: Some(42),
            line_end: Some(45),
            suggestion: Some("Use parameterized queries".into()),
            confidence: 0.9,
            priority_signals: vec![],
        }
    }

    /// Construct a minimal Repository without relying on Default.
    fn test_repo() -> crate::core::models::Repository {
        crate::core::models::Repository {
            owner: "owner".into(),
            name: "repo".into(),
            full_name: "owner/repo".into(),
            description: None,
            language: None,
            languages: HashMap::new(),
            stars: 0,
            forks: 0,
            open_issues: 0,
            topics: vec![],
            default_branch: "main".into(),
            html_url: String::new(),
            clone_url: String::new(),
            has_contributing: false,
            has_license: false,
            last_push_at: None,
            created_at: None,
        }
    }

    /// Construct a minimal RepoContext for tests.
    fn test_context(files: HashMap<String, String>) -> RepoContext {
        RepoContext {
            repo: test_repo(),
            relevant_files: files,
            file_tree: vec![],
            readme_content: None,
            contributing_guide: None,
            open_issues: vec![],
            coding_style: None,
            symbol_map: HashMap::new(),
            file_ranks: HashMap::new(),
        }
    }

    fn mock_gen() -> ContributionGenerator<'static> {
        use std::sync::OnceLock;
        static CONFIG: OnceLock<ContributionConfig> = OnceLock::new();
        let config = CONFIG.get_or_init(ContributionConfig::default);
        static MOCK: MockLlm = MockLlm;
        ContributionGenerator {
            llm: &MOCK,
            config,
            self_review_enabled: false,
        }
    }

    // ── Branch name ───────────────────────────────────────────────────────────

    #[test]
    fn test_generate_branch_name() {
        let f = test_finding();
        let branch = ContributionGenerator::generate_branch_name(&f);
        assert!(branch.starts_with("contribai/fix/security/"));
        assert!(branch.contains("sql-injection"));
    }

    // ── PR title ──────────────────────────────────────────────────────────────

    #[test]
    fn test_generate_pr_title() {
        let f = test_finding();
        let title = ContributionGenerator::generate_pr_title(&f);
        assert!(title.starts_with("Security: "));
    }

    #[test]
    fn test_generate_pr_title_with_conventional_guidelines() {
        let g = RepoGuidelines {
            uses_conventional_commits: true,
            contributing_md: "uses conventional commits".into(),
            pr_template: "## Description".into(),
            ..Default::default()
        };
        let f = test_finding();
        let title = ContributionGenerator::generate_pr_title_with_guidelines(&f, Some(&g));
        // Conventional commits format: "fix: sql injection in user query"
        assert!(title.starts_with("fix:") || title.contains("sql injection"));
    }

    // ── Commit message ────────────────────────────────────────────────────────

    #[test]
    fn test_generate_commit_message() {
        let gen = mock_gen();
        let f = test_finding();
        let changes = vec![FileChange {
            path: "src/db/queries.py".into(),
            original_content: None,
            new_content: "fixed".into(),
            is_new_file: false,
            is_deleted: false,
        }];
        let msg = gen.generate_commit_message(&f, &changes);
        // Should contain "fix(security)" and scope "(db)"
        assert!(msg.contains("fix(security)"));
        assert!(msg.contains("(db)"));
    }

    // ── parse_changes: legacy bare-array backward compat ─────────────────────

    #[test]
    fn test_parse_changes_valid() {
        let gen = mock_gen();
        let ctx = test_context(HashMap::new());
        let response = r#"[{"path": "src/main.py", "new_content": "print('fixed')", "is_new_file": false}]"#;
        let changes = gen.parse_changes(response, &ctx);
        assert!(changes.is_some());
        assert_eq!(changes.unwrap().len(), 1);
    }

    #[test]
    fn test_parse_changes_invalid() {
        let gen = mock_gen();
        let ctx = test_context(HashMap::new());
        let response = "This is not valid JSON at all";
        let changes = gen.parse_changes(response, &ctx);
        assert!(changes.is_none());
    }

    // ── parse_changes: new search/replace format ──────────────────────────────

    #[test]
    fn test_parse_changes_search_replace() {
        let gen = mock_gen();
        let mut files = HashMap::new();
        files.insert(
            "src/main.py".to_string(),
            "def foo():\n    x = 1\n    return x\n".to_string(),
        );
        let ctx = test_context(files);

        let response = r#"{"changes": [{"path": "src/main.py", "is_new_file": false, "edits": [{"search": "x = 1", "replace": "x = 2"}]}]}"#;
        let changes = gen.parse_changes(response, &ctx);
        assert!(changes.is_some());
        let ch = changes.unwrap();
        assert_eq!(ch.len(), 1);
        assert!(ch[0].new_content.contains("x = 2"));
        assert!(!ch[0].new_content.contains("x = 1"));
    }

    // ── extract_json ──────────────────────────────────────────────────────────

    #[test]
    fn test_extract_json_fenced() {
        let response = "some text\n```json\n{\"changes\": []}\n```\ntrailing text";
        let result = ContributionGenerator::extract_json(response);
        assert_eq!(result, Some("{\"changes\": []}".to_string()));
    }

    #[test]
    fn test_extract_json_raw() {
        let response = r#"Here is the fix: {"changes": [{"path": "x.py"}]}"#;
        let result = ContributionGenerator::extract_json(response);
        assert!(result.is_some());
        assert!(result.unwrap().contains("changes"));
    }

    #[test]
    fn test_extract_json_bare_array() {
        let response = r#"[{"path": "x.py", "new_content": "hello"}]"#;
        let result = ContributionGenerator::extract_json(response);
        assert!(result.is_some());
        assert!(result.unwrap().starts_with('['));
    }

    #[test]
    fn test_extract_json_none() {
        let result = ContributionGenerator::extract_json("no json here at all");
        assert!(result.is_none());
    }

    // ── fuzzy_replace ─────────────────────────────────────────────────────────

    #[test]
    fn test_fuzzy_replace_exact_words() {
        let content = "line one\nline two\nline three\n";
        let search = "line one\nline two";
        let replace = "replaced one\nreplaced two";
        let result = ContributionGenerator::fuzzy_replace(content, search, replace);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.contains("replaced one"));
        assert!(!r.contains("line one"));
    }

    #[test]
    fn test_fuzzy_replace_no_match() {
        // Words share no overlap with content
        let content = "completely different text here";
        let search = "foo bar baz qux quux corge grault garply";
        let replace = "something";
        let result = ContributionGenerator::fuzzy_replace(content, search, replace);
        assert!(result.is_none());
    }

    #[test]
    fn test_fuzzy_replace_empty_search() {
        let result = ContributionGenerator::fuzzy_replace("hello world", "", "replacement");
        assert!(result.is_none());
    }

    // ── count_unbalanced_brackets ─────────────────────────────────────────────

    #[test]
    fn test_count_unbalanced_balanced() {
        let code = "fn foo() { let x = (1 + 2); }";
        assert_eq!(ContributionGenerator::count_unbalanced_brackets(code), 0);
    }

    #[test]
    fn test_count_unbalanced_simple_imbalance() {
        // missing `)` and `}`
        let code = "fn foo() { let x = (1 + 2;";
        assert_eq!(ContributionGenerator::count_unbalanced_brackets(code), 2);
    }

    #[test]
    fn test_count_unbalanced_ignores_string() {
        // Brackets inside string literals must be ignored
        let code = r#"let s = "hello { world }"; fn foo() {}"#;
        assert_eq!(ContributionGenerator::count_unbalanced_brackets(code), 0);
    }

    #[test]
    fn test_count_unbalanced_ignores_line_comment_slash() {
        let code = "let x = 1; // unmatched { bracket\nlet y = 2;";
        assert_eq!(ContributionGenerator::count_unbalanced_brackets(code), 0);
    }

    #[test]
    fn test_count_unbalanced_ignores_line_comment_hash() {
        let code = "x = 1  # unmatched { bracket\ny = 2";
        assert_eq!(ContributionGenerator::count_unbalanced_brackets(code), 0);
    }

    #[test]
    fn test_count_unbalanced_ignores_block_comment() {
        let code = "let x = 1; /* unmatched { */ let y = 2;";
        assert_eq!(ContributionGenerator::count_unbalanced_brackets(code), 0);
    }

    // ── validate_changes ──────────────────────────────────────────────────────

    #[test]
    fn test_validate_changes_good() {
        let gen = mock_gen();
        let good = vec![FileChange {
            path: "test.py".into(),
            original_content: None,
            new_content: "def foo():\n    return 42\n".into(),
            is_new_file: false,
            is_deleted: false,
        }];
        assert!(gen.validate_changes(&good));
    }

    #[test]
    fn test_validate_changes_noop() {
        let gen = mock_gen();
        let noop = vec![FileChange {
            path: "test.py".into(),
            original_content: Some("same content".into()),
            new_content: "same content".into(),
            is_new_file: false,
            is_deleted: false,
        }];
        assert!(!gen.validate_changes(&noop));
    }

    #[test]
    fn test_validate_changes_new_file_empty() {
        let gen = mock_gen();
        let empty_new = vec![FileChange {
            path: "test.py".into(),
            original_content: None,
            new_content: "   ".into(),
            is_new_file: true,
            is_deleted: false,
        }];
        assert!(!gen.validate_changes(&empty_new));
    }

    // ── cross-file detection ──────────────────────────────────────────────────

    #[test]
    fn test_find_cross_file_instances() {
        let gen = mock_gen();
        let finding = Finding {
            description: "Use `parameterized queries` instead of `string.format`".into(),
            suggestion: Some("Use `cursor.execute` with params".into()),
            ..test_finding()
        };

        let mut files = HashMap::new();
        // Primary file (should be excluded from results)
        files.insert(
            "src/db/queries.py".to_string(),
            "cursor.execute(sql.format(user))".to_string(),
        );
        // Should match: has 2+ keyword hits
        files.insert(
            "src/api/users.py".to_string(),
            "parameterized queries string.format cursor.execute".to_string(),
        );
        // Should NOT match: insufficient keyword overlap
        files.insert(
            "src/api/posts.py".to_string(),
            "unrelated content here xyz".to_string(),
        );

        let ctx = test_context(files);
        let result = gen.find_cross_file_instances(&finding, &ctx);

        assert!(result.contains_key("src/api/users.py"));
        assert!(!result.contains_key("src/db/queries.py")); // primary file excluded
        assert!(!result.contains_key("src/api/posts.py")); // too few matches
    }

    // ── word overlap ratio ────────────────────────────────────────────────────

    #[test]
    fn test_word_overlap_ratio_identical() {
        let words: Vec<&str> = vec!["foo", "bar", "baz"];
        let ratio = word_overlap_ratio(&words, &words);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_word_overlap_ratio_disjoint() {
        let a = vec!["foo", "bar"];
        let b = vec!["qux", "quux"];
        let ratio = word_overlap_ratio(&a, &b);
        assert!((ratio - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_word_overlap_ratio_partial() {
        let a = vec!["foo", "bar", "baz"];
        let b = vec!["foo", "bar", "qux"];
        let ratio = word_overlap_ratio(&a, &b);
        // intersection = 2, total = 6 → 2*2/6 ≈ 0.667
        assert!(ratio > 0.5 && ratio < 1.0);
    }

    // ── Mock LLM ──────────────────────────────────────────────────────────────

    struct MockLlm;

    #[async_trait::async_trait]
    impl LlmProvider for MockLlm {
        async fn complete(
            &self,
            _prompt: &str,
            _system: Option<&str>,
            _temperature: Option<f64>,
            _max_tokens: Option<u32>,
        ) -> Result<String> {
            Ok("mock response".into())
        }

        async fn chat(
            &self,
            _messages: &[crate::llm::provider::ChatMessage],
            _system: Option<&str>,
            _temperature: Option<f64>,
            _max_tokens: Option<u32>,
        ) -> Result<String> {
            Ok("mock response".into())
        }
    }
}
