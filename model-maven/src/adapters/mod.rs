mod artifacts;
pub mod component_scaffold;
pub mod components;
pub mod exports;
mod preview_repository;
pub mod previews;
pub mod project_config;
pub mod project_source;
pub mod python;
pub mod runs;
pub mod source_edits;
pub mod supervisor;
pub mod variants;
mod worker_assets;

/// Flush directory metadata where the OS supports opening directories as files.
/// Windows does not expose a portable directory fsync; file contents are still
/// flushed before atomic replacement, but power-loss durability is weaker.
pub(crate) fn sync_directory(path: &std::path::Path) -> std::io::Result<()> {
    #[cfg(not(windows))]
    {
        std::fs::File::open(path)?.sync_all()
    }
    #[cfg(windows)]
    {
        if std::fs::metadata(path)?.is_dir() {
            Ok(())
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "not a directory"))
        }
    }
}
