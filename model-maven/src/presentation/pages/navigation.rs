use crate::presentation::{
    contracts::{NavigationResult, PageDescriptor},
    session::SessionState,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Selection {
    Navigate(NavigationResult),
    Ambiguous(Vec<String>),
    Invalid,
}

pub fn resolve_selection(page: &PageDescriptor, input: &str) -> Selection {
    let query = input.trim();
    if query.is_empty() {
        return Selection::Navigate(NavigationResult::Stay);
    }
    if query == "0" {
        return Selection::Navigate(NavigationResult::Quit);
    }
    if matches!(query.to_ascii_lowercase().as_str(), "back" | "b" | "返回") {
        return Selection::Navigate(NavigationResult::Back);
    }
    if matches!(query.to_ascii_lowercase().as_str(), "quit" | "q" | "退出") {
        return Selection::Navigate(NavigationResult::Quit);
    }

    let selected = query.parse::<usize>().ok().and_then(|index| {
        index
            .checked_sub(1)
            .and_then(|index| page.actions.get(index))
    });
    let selected = selected.or_else(|| {
        page.actions
            .iter()
            .find(|action| action.label.eq_ignore_ascii_case(query))
    });
    if let Some(action) = selected {
        return action_selection(action.id.as_str());
    }

    let query = query.strip_prefix('/').unwrap_or(query).to_lowercase();
    let matches = page
        .actions
        .iter()
        .filter(|action| action.label.to_lowercase().contains(&query))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [action] => action_selection(action.id.as_str()),
        [] => Selection::Invalid,
        _ => Selection::Ambiguous(matches.iter().map(|action| action.label.clone()).collect()),
    }
}

fn action_selection(id: &str) -> Selection {
    if let Some(page_id) = id.strip_prefix("page:") {
        Selection::Navigate(NavigationResult::Open(
            crate::presentation::contracts::PageId::new(page_id),
        ))
    } else if matches!(id, "back" | "quit") {
        Selection::Navigate(if id == "quit" {
            NavigationResult::Quit
        } else {
            NavigationResult::Back
        })
    } else {
        Selection::Navigate(NavigationResult::RunAction(id.to_owned()))
    }
}

pub fn go_back(session: &mut SessionState) {
    session.navigate(NavigationResult::Back);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presentation::{
        contracts::{ActionDescriptor, PageDescriptor, PageId},
        pages::home::register_pages,
        session::SessionState,
    };

    #[test]
    fn home_navigation_accepts_labels_search_and_back_without_exposing_action_ids() {
        let pages = register_pages("project", None, None, None, None, "没有活动运行。").unwrap();
        let home = pages.get(&PageId::new("home")).unwrap();
        let Selection::Navigate(NavigationResult::Open(target)) =
            resolve_selection(home, "模型与 Layer")
        else {
            panic!("expected model page")
        };
        assert_eq!(target.0, "model");
        let Selection::Navigate(NavigationResult::Open(target)) = resolve_selection(home, "/训练")
        else {
            panic!("expected a unique keyword match")
        };
        assert_eq!(target.0, "runs");
        assert_eq!(
            resolve_selection(home, "0"),
            Selection::Navigate(NavigationResult::Quit)
        );

        let mut session = SessionState::new(PageId::new("home"));
        session.navigate(NavigationResult::Open(target));
        go_back(&mut session);
        assert_eq!(session.current_page.0, "home");
    }

    #[test]
    fn ambiguous_actions_are_not_guessed() {
        let page = PageDescriptor {
            id: PageId::new("x"),
            title: String::new(),
            summary: String::new(),
            actions: vec![
                ActionDescriptor::available("a", "查看模型"),
                ActionDescriptor::available("b", "查看运行"),
            ],
        };
        assert!(matches!(
            resolve_selection(&page, "查看"),
            Selection::Ambiguous(_)
        ));
    }
}
