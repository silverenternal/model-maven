use std::{env, fs, io, path::{Path, PathBuf}};

fn collect(root: &Path, directory: &Path, paths: &mut Vec<PathBuf>) -> io::Result<()> {
    println!("cargo:rerun-if-changed={}", directory.display());
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_symlink() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("workspace contains symlink: {}", path.display())));
        }
        if kind.is_dir() {
            collect(root, &path, paths)?;
        } else if kind.is_file() && path.extension().is_some_and(|extension| extension == "lua") {
            paths.push(path.strip_prefix(root).map_err(io::Error::other)?.to_path_buf());
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or_else(|| io::Error::other("missing manifest directory"))?);
    let mut paths = Vec::new();
    collect(&root, &root.join("lua/model_maven"), &mut paths)?;
    collect(&root, &root.join("plugin"), &mut paths)?;
    paths.sort();
    for required in ["lua/model_maven/init.lua", "lua/model_maven/workspace.lua", "plugin/model_maven.lua"] {
        if !paths.iter().any(|path| path == Path::new(required)) {
            return Err(io::Error::new(io::ErrorKind::NotFound, format!("missing workspace entrypoint: {required}")));
        }
    }
    let mut generated = String::from("const WORKSPACE_SOURCES: &[(&str, &str)] = &[\n");
    for path in paths {
        let relative = path.to_str().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "workspace path is not UTF-8"))?.replace('\\', "/");
        println!("cargo:rerun-if-changed={}", root.join(&path).display());
        generated.push_str(&format!("    ({relative:?}, include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), {suffix:?}))),\n", suffix = format!("/{relative}")));
    }
    generated.push_str("];\n");
    let output = PathBuf::from(env::var_os("OUT_DIR").ok_or_else(|| io::Error::other("missing output directory"))?);
    fs::write(output.join("workspace_sources.rs"), generated)
}
