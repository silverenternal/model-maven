use crate::ports::ProjectSourceWriter;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

pub struct FileProjectSourceWriter;

impl ProjectSourceWriter for FileProjectSourceWriter {
    fn write_new(
        &self,
        root: &Path,
        relative: &Path,
        source: &[u8],
    ) -> Result<PathBuf, (&'static str, String)> {
        let mut parent = root.to_path_buf();
        if let Some(parent_relative) = relative.parent() {
            for component in parent_relative.components() {
                let std::path::Component::Normal(name) = component else {
                    return Err((
                        "ASSEMBLY_OUTPUT_PATH_INVALID",
                        "输出目录包含不安全路径部分。".into(),
                    ));
                };
                parent.push(name);
                match fs::symlink_metadata(&parent) {
                    Ok(metadata) if metadata.file_type().is_symlink() => {
                        return Err((
                            "ASSEMBLY_OUTPUT_PATH_INVALID",
                            format!("生成目录不能经过符号链接：{}。", parent.display()),
                        ));
                    }
                    Ok(metadata) if !metadata.is_dir() => {
                        return Err((
                            "ASSEMBLY_OUTPUT_PATH_INVALID",
                            format!("输出父路径不是目录：{}。", parent.display()),
                        ));
                    }
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        fs::create_dir(&parent).map_err(|create_error| {
                            (
                                "ASSEMBLY_OUTPUT_WRITE_FAILED",
                                format!("无法创建生成目录 {}：{create_error}", parent.display()),
                            )
                        })?;
                    }
                    Err(error) => {
                        return Err((
                            "ASSEMBLY_OUTPUT_WRITE_FAILED",
                            format!("无法检查生成目录 {}：{error}", parent.display()),
                        ));
                    }
                }
                let canonical = parent.canonicalize().map_err(|error| {
                    (
                        "ASSEMBLY_OUTPUT_WRITE_FAILED",
                        format!("无法验证生成目录 {}：{error}", parent.display()),
                    )
                })?;
                if !canonical.starts_with(root) {
                    return Err((
                        "ASSEMBLY_OUTPUT_PATH_INVALID",
                        "输出目录不能离开当前项目根目录。".into(),
                    ));
                }
            }
        }
        let target = root.join(relative);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)
            .map_err(|error| {
                let code = if error.kind() == std::io::ErrorKind::AlreadyExists {
                    "ASSEMBLY_OUTPUT_EXISTS"
                } else {
                    "ASSEMBLY_OUTPUT_WRITE_FAILED"
                };
                (
                    code,
                    format!("无法创建生成文件 {}：{error}", target.display()),
                )
            })?;
        if let Err(error) = file.write_all(source).and_then(|()| file.sync_all()) {
            drop(file);
            let _ = fs::remove_file(&target);
            return Err((
                "ASSEMBLY_OUTPUT_WRITE_FAILED",
                format!("生成文件写入失败，已清理不完整输出：{error}"),
            ));
        }
        Ok(target)
    }
}
