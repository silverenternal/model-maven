use super::contracts::{NavigationResult, PageId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionState {
    pub current_page: PageId,
    history: Vec<PageId>,
    pub model: Option<String>,
    pub model_source: Option<String>,
    pub batch: Option<String>,
    pub variant: Option<String>,
    quit_requested: bool,
}

impl SessionState {
    pub fn new(home: PageId) -> Self {
        Self {
            current_page: home,
            history: Vec::new(),
            model: None,
            model_source: None,
            batch: None,
            variant: None,
            quit_requested: false,
        }
    }

    pub fn navigate(&mut self, result: NavigationResult) {
        match result {
            NavigationResult::Stay => {}
            NavigationResult::Open(page) => {
                self.history.push(self.current_page.clone());
                self.current_page = page;
            }
            NavigationResult::RunAction(_) => {}
            NavigationResult::Back => {
                if let Some(page) = self.history.pop() {
                    self.current_page = page;
                }
            }
            NavigationResult::Quit => self.quit_requested = true,
        }
    }

    pub fn can_go_back(&self) -> bool {
        !self.history.is_empty()
    }

    pub fn quit_requested(&self) -> bool {
        self.quit_requested
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn navigation_tracks_back_and_quit_without_losing_session_inputs() {
        let home = PageId::new("home");
        let layers = PageId::new("layers");
        let mut state = SessionState::new(home.clone());
        state.model = Some("project:model".into());
        state.navigate(NavigationResult::Open(layers.clone()));
        assert_eq!(state.current_page, layers);
        assert!(state.can_go_back());
        state.navigate(NavigationResult::Back);
        assert_eq!(state.current_page, home);
        assert!(!state.can_go_back());
        state.navigate(NavigationResult::Back);
        assert_eq!(state.current_page, home);
        state.navigate(NavigationResult::Quit);
        assert!(state.quit_requested());
        assert_eq!(state.model.as_deref(), Some("project:model"));
    }
}
