use serde_json::Value;
use std::collections::BTreeSet;

pub struct ModelTreePage {
    query: String,
    page: usize,
    page_size: usize,
    collapsed: BTreeSet<String>,
}

impl Default for ModelTreePage {
    fn default() -> Self {
        Self {
            query: String::new(),
            page: 0,
            page_size: 40,
            collapsed: BTreeSet::new(),
        }
    }
}

impl ModelTreePage {
    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn page(&self) -> usize {
        self.page
    }

    pub fn set_query(&mut self, query: impl Into<String>) {
        self.query = query.into();
        self.page = 0;
        self.collapsed.clear();
    }

    pub fn next(&mut self, rows: &[Value]) {
        let count = self.visible_rows(rows).len();
        let last = count.saturating_sub(1) / self.page_size;
        self.page = (self.page + 1).min(last);
    }

    pub fn previous(&mut self) {
        self.page = self.page.saturating_sub(1);
    }

    pub fn toggle(&mut self, path: &str) -> bool {
        if self.collapsed.remove(path) {
            false
        } else {
            self.collapsed.insert(path.to_owned());
            true
        }
    }

    pub fn visible_rows<'a>(&self, rows: &'a [Value]) -> Vec<&'a Value> {
        let matches = rows
            .iter()
            .filter(|row| self.matches(row))
            .filter_map(|row| row["path"].as_str())
            .collect::<BTreeSet<_>>();
        let included = if self.query.trim().is_empty() {
            rows.iter()
                .filter_map(|row| row["path"].as_str())
                .collect::<BTreeSet<_>>()
        } else {
            let mut included = BTreeSet::new();
            for path in matches {
                included.insert(path);
                let mut parent = path;
                while let Some((prefix, _)) = parent.rsplit_once('.') {
                    included.insert(prefix);
                    parent = prefix;
                }
            }
            included
        };

        rows.iter()
            .filter(|row| {
                let Some(path) = row["path"].as_str() else {
                    return false;
                };
                if !included.contains(path) {
                    return false;
                }
                !self
                    .collapsed
                    .iter()
                    .any(|parent| path != parent && path.starts_with(&format!("{parent}.")))
            })
            .collect()
    }

    pub fn current_page<'a>(&self, rows: &'a [Value]) -> Vec<&'a Value> {
        let visible = self.visible_rows(rows);
        let start = self.page.saturating_mul(self.page_size).min(visible.len());
        let end = start.saturating_add(self.page_size).min(visible.len());
        visible[start..end].to_vec()
    }

    pub fn page_count(&self, rows: &[Value]) -> usize {
        self.visible_rows(rows)
            .len()
            .div_ceil(self.page_size)
            .max(1)
    }

    pub fn match_count(&self, rows: &[Value]) -> usize {
        rows.iter().filter(|row| self.matches(row)).count()
    }

    fn matches(&self, row: &Value) -> bool {
        let query = self.query.trim().to_lowercase();
        if query.is_empty() {
            return true;
        }
        ["path", "type", "source"]
            .into_iter()
            .filter_map(|key| row[key].as_str())
            .any(|value| value.to_lowercase().contains(&query))
            || row["aliases"].as_array().is_some_and(|aliases| {
                aliases
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|a| a.to_lowercase().contains(&query))
            })
            || row["editable_params"].as_object().is_some_and(|params| {
                params
                    .keys()
                    .any(|name| name.to_lowercase().contains(&query))
            })
            || row["source_editable_fields"]
                .as_array()
                .is_some_and(|fields| {
                    fields
                        .iter()
                        .filter_map(Value::as_str)
                        .any(|name| name.to_lowercase().contains(&query))
                })
            || ["input_shape", "output_shape", "inputs", "outputs"]
                .into_iter()
                .filter_map(|key| row.get(key))
                .filter(|value| !value.is_null())
                .any(|value| value.to_string().to_lowercase().contains(&query))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerSelection {
    pub path: String,
    pub instance_id: String,
}

impl LayerSelection {
    pub fn from_row(row: &Value) -> Option<Self> {
        Some(Self {
            path: row["path"].as_str()?.to_owned(),
            instance_id: row["instance_id"].as_str()?.to_owned(),
        })
    }

    /// A prior selection stays valid only when both its path and module identity match.
    pub fn still_points_to_same_layer(&self, row: &Value) -> bool {
        row["path"].as_str() == Some(self.path.as_str())
            && row["instance_id"].as_str() == Some(self.instance_id.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rows() -> Vec<Value> {
        vec![
            json!({"path":"model","type":"Sequential","instance_id":"m0","parameters":8,"can_insert":true}),
            json!({"path":"model.encoder","type":"ModuleDict","instance_id":"m1","parameters":8,"can_insert":true}),
            json!({"path":"model.encoder.proj","type":"Linear","instance_id":"shared","aliases":["model.encoder.proj","model.decoder.proj"],"own_parameters":8,"editable_params":{"out_features":4}}),
            json!({"path":"model.decoder","type":"Sequential","instance_id":"m2","parameters":8,"can_insert":true}),
            json!({"path":"model.decoder.proj","type":"Linear","instance_id":"shared","aliases":["model.encoder.proj","model.decoder.proj"],"own_parameters":8,"editable_params":{"out_features":4}}),
        ]
    }

    #[test]
    fn search_matches_type_alias_and_fields_and_keeps_ancestors() {
        let rows = rows();
        let mut page = ModelTreePage::default();
        page.set_query("Linear");
        let paths = page
            .visible_rows(&rows)
            .iter()
            .filter_map(|row| row["path"].as_str())
            .collect::<Vec<_>>();
        assert!(paths.contains(&"model"));
        assert!(paths.contains(&"model.encoder"));
        assert!(paths.contains(&"model.decoder.proj"));

        page.set_query("out_features");
        assert_eq!(page.visible_rows(&rows).len(), 5);
        page.set_query("decoder.proj");
        assert!(
            page.visible_rows(&rows)
                .iter()
                .any(|row| row["path"] == "model.encoder.proj")
        );
    }

    #[test]
    fn collapse_and_pagination_bound_long_tree_without_losing_paths() {
        let mut rows = rows();
        for index in 0..1000 {
            rows.push(json!({"path":format!("model.layer_{index}"),"type":"Linear","instance_id":format!("i{index}")}));
        }
        let mut page = ModelTreePage::default();
        assert_eq!(page.page_count(&rows), 26);
        assert_eq!(page.current_page(&rows).len(), 40);
        page.next(&rows);
        assert_eq!(page.page(), 1);
        page.set_query("layer_999");
        assert!(
            page.visible_rows(&rows)
                .iter()
                .any(|row| row["path"] == "model.layer_999")
        );
        page.set_query("");
        page.toggle("model.encoder");
        assert!(
            !page
                .visible_rows(&rows)
                .iter()
                .any(|row| row["path"] == "model.encoder.proj")
        );
    }

    #[test]
    fn selection_is_invalidated_when_path_changes_after_renumbering() {
        let selected =
            LayerSelection::from_row(&json!({"path":"model.layers.2","instance_id":"module-2"}))
                .unwrap();
        assert!(!selected.still_points_to_same_layer(
            &json!({"path":"model.layers.1","instance_id":"module-2"})
        ));
        assert!(!selected.still_points_to_same_layer(
            &json!({"path":"model.layers.2","instance_id":"module-3"})
        ));
        assert!(selected.still_points_to_same_layer(
            &json!({"path":"model.layers.2","instance_id":"module-2"})
        ));
    }
}
