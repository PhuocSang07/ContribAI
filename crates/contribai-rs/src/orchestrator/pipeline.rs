//! Main pipeline orchestrator.
//!
//! Port from Python `orchestrator/pipeline.py`.
//! Coordinates: discover → analyze → generate → PR.

use std::collections::HashSet;
use tracing::{info, warn};

use crate::analysis::analyzer::CodeAnalyzer;
use crate::core::config::ContribAIConfig;
use crate::core::error::Result;
use crate::core::events::{Event, EventBus, EventType};
use crate::core::models::{AnalysisResult, DiscoveryCriteria, Repository};
use crate::generator::engine::ContributionGenerator;
use crate::generator::scorer::QualityScorer;
use crate::github::client::GitHubClient;
use crate::github::discovery::RepoDiscovery;
use crate::llm::provider::LlmProvider;
use crate::orchestrator::memory::Memory;
use crate::pr::manager::PrManager;

/// Files that should NEVER be modified by ContribAI.
const PROTECTED_META_FILES: &[&str] = &[
    "CONTRIBUTING.md",
    ".github/CONTRIBUTING.md",
    "docs/CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    ".github/CODE_OF_CONDUCT.md",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    ".github/FUNDING.yml",
    ".github/SECURITY.md",
    "SECURITY.md",
    ".github/CODEOWNERS",
    ".all-contributorsrc",
];

/// Extensions to skip.
const SKIP_EXTENSIONS: &[&str] = &[
    ".md", ".txt", ".rst", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".json",
];

/// Directories to skip.
const SKIP_DIRECTORIES: &[&str] = &[
    "examples", "example", "samples", "sample", "demos", "demo",
    "docs", "doc", "test", "tests", "testing", "test_data", "testdata",
    "fixtures", "benchmarks", "benchmark", "__pycache__", "vendor",
    "third_party", "third-party", "node_modules",
];

/// Result of a pipeline run.
#[derive(Debug, Clone, Default)]
pub struct PipelineResult {
    pub repos_analyzed: usize,
    pub findings_total: usize,
    pub contributions_generated: usize,
    pub prs_created: usize,
    pub errors: Vec<String>,
}

/// Main orchestrator for the contribution pipeline.
pub struct ContribPipeline<'a> {
    config: &'a ContribAIConfig,
    github: &'a GitHubClient,
    llm: &'a dyn LlmProvider,
    memory: &'a Memory,
    event_bus: &'a EventBus,
    scorer: QualityScorer,
}

impl<'a> ContribPipeline<'a> {
    pub fn new(
        config: &'a ContribAIConfig,
        github: &'a GitHubClient,
        llm: &'a dyn LlmProvider,
        memory: &'a Memory,
        event_bus: &'a EventBus,
    ) -> Self {
        let min_quality = config.pipeline.min_quality_score;
        Self {
            config,
            github,
            llm,
            memory,
            event_bus,
            scorer: QualityScorer::new(min_quality),
        }
    }

    /// Run the full pipeline: discover → analyze → generate → PR.
    pub async fn run(
        &self,
        criteria: Option<&DiscoveryCriteria>,
        dry_run: bool,
    ) -> Result<PipelineResult> {
        let mut result = PipelineResult::default();
        let run_id = self.memory.start_run()?;

        self.event_bus
            .emit(
                Event::new(EventType::PipelineStart, "pipeline.run")
                    .with_data("dry_run", dry_run),
            )
            .await;

        // Check daily PR limit
        let today_prs = self.memory.get_today_pr_count()?;
        let remaining_prs = (self.config.github.max_prs_per_day as usize).saturating_sub(today_prs);
        if remaining_prs == 0 && !dry_run {
            warn!(
                limit = self.config.github.max_prs_per_day,
                "Daily PR limit reached"
            );
            return Ok(result);
        }

        // 1. Discover repos
        let default_criteria = DiscoveryCriteria::default();
        let criteria = criteria.unwrap_or(&default_criteria);

        info!("🔍 Discovering repositories...");
        let discovery = RepoDiscovery::new(self.github, &self.config.discovery);
        let repos = discovery.discover(Some(criteria)).await?;

        if repos.is_empty() {
            warn!("No repositories found matching criteria");
            return Ok(result);
        }

        info!(count = repos.len(), "Found candidate repositories");

        // Limit to max repos per run
        let max_repos = self.config.pipeline.max_repos_per_run;
        let repos: Vec<_> = repos.into_iter().take(max_repos).collect();

        // 2. Process each repo
        for repo in &repos {
            if self.memory.has_analyzed(&repo.full_name)? {
                info!(repo = %repo.full_name, "Skipping (already analyzed)");
                continue;
            }

            match self.process_repo(repo, dry_run, remaining_prs).await {
                Ok(repo_result) => {
                    result.repos_analyzed += 1;
                    result.findings_total += repo_result.findings_total;
                    result.contributions_generated += repo_result.contributions_generated;
                    result.prs_created += repo_result.prs_created;
                    result.errors.extend(repo_result.errors);
                }
                Err(e) => {
                    let msg = format!("Error processing {}: {}", repo.full_name, e);
                    warn!("{}", msg);
                    result.errors.push(msg);
                }
            }
        }

        // 3. Log run
        self.memory.finish_run(
            run_id,
            result.repos_analyzed as i64,
            result.prs_created as i64,
            result.findings_total as i64,
            result.errors.len() as i64,
        )?;

        self.event_bus
            .emit(
                Event::new(EventType::PipelineComplete, "pipeline.run")
                    .with_data("repos", result.repos_analyzed as i64)
                    .with_data("prs", result.prs_created as i64)
                    .with_data("findings", result.findings_total as i64),
            )
            .await;

        Ok(result)
    }

    /// Process a single repository.
    async fn process_repo(
        &self,
        repo: &Repository,
        dry_run: bool,
        max_prs: usize,
    ) -> Result<PipelineResult> {
        let mut result = PipelineResult::default();

        self.event_bus
            .emit(
                Event::new(EventType::AnalysisStart, "pipeline.process_repo")
                    .with_data("repo", repo.full_name.as_str()),
            )
            .await;

        info!(repo = %repo.full_name, "📦 Processing");

        // 1. Analyze
        let analyzer = CodeAnalyzer::new(self.llm, self.github, &self.config.analysis);
        let analysis = analyzer.analyze(repo).await?;

        result.findings_total = analysis.findings.len();

        self.memory.record_analysis(
            &repo.full_name,
            repo.language.as_deref().unwrap_or("unknown"),
            repo.stars,
            analysis.findings.len() as i64,
        )?;

        self.event_bus
            .emit(
                Event::new(EventType::AnalysisComplete, "pipeline.process_repo")
                    .with_data("repo", repo.full_name.as_str())
                    .with_data("findings", analysis.findings.len() as i64),
            )
            .await;

        if analysis.findings.is_empty() {
            info!(repo = %repo.full_name, "✅ No findings");
            return Ok(result);
        }

        // Filter findings
        let findings = self.filter_findings(&analysis);
        info!(
            repo = %repo.full_name,
            raw = analysis.findings.len(),
            filtered = findings.len(),
            "Findings filtered"
        );

        if findings.is_empty() {
            return Ok(result);
        }

        // 2. Generate contributions
        let generator = ContributionGenerator::new(self.llm, &self.config.contribution);

        let context = self.build_repo_context(repo, &analysis).await;

        for finding in findings.iter().take(max_prs) {
            self.event_bus
                .emit(
                    Event::new(EventType::GenerationStart, "pipeline.process_repo")
                        .with_data("title", finding.title.as_str()),
                )
                .await;

            match generator.generate(finding, &context).await {
                Ok(Some(contribution)) => {
                    // Quality check
                    let report = self.scorer.evaluate(&contribution);
                    if !report.passed {
                        info!(
                            title = %contribution.title,
                            score = report.score,
                            "❌ Quality check failed"
                        );
                        continue;
                    }

                    result.contributions_generated += 1;

                    if dry_run {
                        info!(
                            title = %contribution.title,
                            score = report.score,
                            "🏃 [DRY RUN] Would create PR"
                        );
                    } else {
                        // Create PR
                        let mut pr_mgr = PrManager::new(self.github);
                        match pr_mgr.create_pr(&contribution, repo).await {
                            Ok(pr_result) => {
                                result.prs_created += 1;
                                self.memory.record_pr(
                                    &repo.full_name,
                                    pr_result.pr_number,
                                    &pr_result.pr_url,
                                    &contribution.title,
                                    &contribution.contribution_type.to_string(),
                                    &pr_result.branch_name,
                                    &pr_result.fork_full_name,
                                )?;

                                self.event_bus
                                    .emit(
                                        Event::new(EventType::PrCreated, "pipeline.process_repo")
                                            .with_data("pr_number", pr_result.pr_number)
                                            .with_data("url", pr_result.pr_url.as_str()),
                                    )
                                    .await;
                            }
                            Err(e) => {
                                let msg = format!("PR creation failed: {}", e);
                                warn!("{}", msg);
                                result.errors.push(msg);
                            }
                        }
                    }
                }
                Ok(None) => {
                    info!(title = %finding.title, "No contribution generated");
                }
                Err(e) => {
                    result.errors.push(format!("Generation error: {}", e));
                }
            }
        }

        Ok(result)
    }

    /// Filter findings: remove protected files, skip extensions, dedup.
    fn filter_findings(
        &self,
        analysis: &AnalysisResult,
    ) -> Vec<crate::core::models::Finding> {
        let protected: HashSet<&str> = PROTECTED_META_FILES.iter().copied().collect();

        analysis
            .findings
            .iter()
            .filter(|f| {
                // Skip protected files
                if protected.contains(f.file_path.as_str()) {
                    return false;
                }

                // Skip by extension
                if let Some(ext) = std::path::Path::new(&f.file_path).extension() {
                    let ext_str = format!(".{}", ext.to_string_lossy().to_lowercase());
                    if SKIP_EXTENSIONS.contains(&ext_str.as_str()) {
                        return false;
                    }
                }

                // Skip directories
                let parts: Vec<&str> = f.file_path.split('/').collect();
                if parts.iter().any(|p| SKIP_DIRECTORIES.contains(p)) {
                    return false;
                }

                true
            })
            .cloned()
            .collect()
    }

    /// Build a RepoContext for the generator.
    async fn build_repo_context(
        &self,
        repo: &Repository,
        _analysis: &AnalysisResult,
    ) -> crate::core::models::RepoContext {
        // Try to get cached style from working memory
        let coding_style = self.memory
            .get_context(&repo.full_name, "coding_style")
            .ok()
            .flatten();

        crate::core::models::RepoContext {
            repo: repo.clone(),
            file_tree: Vec::new(),
            readme_content: None,
            contributing_guide: None,
            relevant_files: std::collections::HashMap::new(),
            open_issues: Vec::new(),
            coding_style,
            symbol_map: std::collections::HashMap::new(),
            file_ranks: std::collections::HashMap::new(),
        }
    }
}

/// Check if two titles are similar (>50% keyword overlap).
pub fn titles_similar(title_a: &str, title_b: &str) -> bool {
    let stop_words: HashSet<&str> = ["a", "an", "the", "in", "on", "of", "for", "to", "and", "or", "is"]
        .iter()
        .copied()
        .collect();

    let words_a: HashSet<String> = title_a
        .to_lowercase()
        .split_whitespace()
        .filter(|w| !stop_words.contains(w) && w.len() > 2)
        .map(String::from)
        .collect();

    let words_b: HashSet<String> = title_b
        .to_lowercase()
        .split_whitespace()
        .filter(|w| !stop_words.contains(w) && w.len() > 2)
        .map(String::from)
        .collect();

    if words_a.is_empty() || words_b.is_empty() {
        return false;
    }

    let overlap = words_a.intersection(&words_b).count();
    let smaller = words_a.len().min(words_b.len());
    overlap as f64 / smaller as f64 > 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_titles_similar() {
        assert!(titles_similar(
            "SQL injection vulnerability fix",
            "SQL injection vulnerability"
        ));
        assert!(!titles_similar(
            "Add logging middleware",
            "Fix database connection pooling"
        ));
    }

    #[test]
    fn test_titles_similar_empty() {
        assert!(!titles_similar("", "something"));
        assert!(!titles_similar("a", "b")); // too short
    }

    #[test]
    fn test_protected_files() {
        let protected: HashSet<&str> = PROTECTED_META_FILES.iter().copied().collect();
        assert!(protected.contains("CONTRIBUTING.md"));
        assert!(protected.contains("LICENSE"));
        assert!(!protected.contains("src/main.py"));
    }

    #[test]
    fn test_skip_directories() {
        assert!(SKIP_DIRECTORIES.contains(&"test"));
        assert!(SKIP_DIRECTORIES.contains(&"vendor"));
        assert!(!SKIP_DIRECTORIES.contains(&"src"));
    }
}
