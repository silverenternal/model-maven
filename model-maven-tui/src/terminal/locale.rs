#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Locale {
    English,
    Simplified,
    Traditional,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum LanguageChoice {
    #[default]
    System,
    En,
    ZhCn,
    ZhTw,
}

impl LanguageChoice {
    pub(crate) fn resolve(self) -> Locale {
        match self {
            Self::System => Locale::current(),
            Self::En => Locale::English,
            Self::ZhCn => Locale::Simplified,
            Self::ZhTw => Locale::Traditional,
        }
    }

    pub(crate) fn label(self, locale: Locale) -> &'static str {
        match self {
            Self::System => locale.text("Follow system", "跟随系统", "跟隨系統"),
            Self::En => "English",
            Self::ZhCn => "简体中文",
            Self::ZhTw => "繁體中文",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "0" | "system" | "auto" => Some(Self::System),
            "1" | "en" => Some(Self::En),
            "2" | "zh-cn" | "zh-hans" => Some(Self::ZhCn),
            "3" | "zh-tw" | "zh-hant" => Some(Self::ZhTw),
            _ => None,
        }
    }
}

impl Locale {
    pub(crate) fn language_label(self) -> &'static str {
        match self {
            Self::English => "English",
            Self::Simplified => "简体中文",
            Self::Traditional => "繁體中文",
        }
    }

    pub(crate) fn current() -> Self {
        let setting = ["LC_ALL", "LC_MESSAGES", "LANG"]
            .into_iter()
            .find_map(|key| {
                std::env::var(key)
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            });
        Self::from_tag(setting.as_deref().unwrap_or_default())
    }

    pub(crate) fn from_tag(tag: &str) -> Self {
        let tag = tag.trim().split(['.', '@']).next().unwrap_or_default();
        let mut parts = tag.split(['-', '_']);
        if !parts
            .next()
            .is_some_and(|part| part.eq_ignore_ascii_case("zh"))
        {
            return Self::English;
        }
        let mut locale = Self::Simplified;
        for part in parts {
            match part.to_ascii_lowercase().as_str() {
                "hans" => return Self::Simplified,
                "hant" => return Self::Traditional,
                "tw" | "hk" | "mo" => locale = Self::Traditional,
                _ => {}
            }
        }
        locale
    }

    pub(crate) fn text(self, en: &'static str, zh: &'static str, tw: &'static str) -> &'static str {
        match self {
            Self::English => en,
            Self::Simplified => zh,
            Self::Traditional => tw,
        }
    }
}
