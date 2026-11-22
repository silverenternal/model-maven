use std::path::{Path, PathBuf};

/// Creates readable generated source within a project without overwriting files.
pub trait ProjectSourceWriter {
    fn write_new(
        &self,
        root: &Path,
        relative: &Path,
        source: &[u8],
    ) -> Result<PathBuf, (&'static str, String)>;
}
