use crate::presentation::contracts::{ActionDescriptor, PageDescriptor, PageId};
use std::path::{Path, PathBuf};

pub fn render(current_path: &str, recognized: bool) -> PageDescriptor {
    let status = if recognized {
        "当前目录包含项目默认模型入口。打开后会使用该项目的配置与 Python 环境。"
    } else {
        "当前目录没有已识别的 Model-Maven 默认入口；你仍可从这里接入 Python 模型。"
    };
    PageDescriptor {
        id: PageId::new("project_open"),
        title: "打开项目".into(),
        summary: format!(
            "当前目录：{current_path}\n{status}\n输入目标目录；路径必须存在且为目录。打开项目只切换工作目录，不改写项目文件。"
        ),
        actions: vec![
            ActionDescriptor::available("open_path", "选择目录"),
            ActionDescriptor::available("back", "返回"),
        ],
    }
}

pub fn resolve_path(input: &str, current: &Path) -> Result<PathBuf, String> {
    let value = input.trim();
    if value.is_empty() {
        return Err("请输入项目目录；没有切换当前项目。".into());
    }
    let path = Path::new(value);
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        current.join(path)
    };
    let canonical = candidate
        .canonicalize()
        .map_err(|error| format!("无法打开目录 {}：{error}", candidate.display()))?;
    if !canonical.is_dir() {
        return Err(format!("目标不是目录：{}", canonical.display()));
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::resolve_path;
    use std::path::Path;

    #[test]
    fn project_path_resolves_relative_to_the_current_directory_and_rejects_files() {
        let root =
            std::env::temp_dir().join(format!("model-maven-project-open-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("project")).unwrap();
        std::fs::write(root.join("file"), "not a directory").unwrap();
        assert_eq!(
            resolve_path("project", &root).unwrap(),
            root.join("project").canonicalize().unwrap()
        );
        assert!(resolve_path("file", &root).is_err());
        assert!(resolve_path("", &root).is_err());
        let _ = std::fs::remove_dir_all(root);
        assert!(Path::new(".").is_dir());
    }
}
