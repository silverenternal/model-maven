//! The preview's visual presets. Shape evidence and layout stay in the scene.
use crate::{
    terminal::{layout::Background, locale::Locale},
    workbench::Style,
};

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum LinkInk {
    #[default]
    Structure,
    Focus,
    FocusBright,
    Unknown,
    Match,
    Features,
    Mismatch,
    Draft,
    Confirmed,
}

impl From<super::Alignment> for LinkInk {
    fn from(alignment: super::Alignment) -> Self {
        match alignment {
            super::Alignment::Match => Self::Match,
            super::Alignment::Features => Self::Features,
            super::Alignment::Mismatch => Self::Mismatch,
            super::Alignment::Unknown => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Appearance {
    pub style: Style,
    pub background: Background,
}

impl Appearance {
    pub fn divider(self) -> char {
        match self.style {
            Style::Default => '─',
            Style::Compact => '┄',
            Style::Story => '━',
        }
    }

    pub fn focus(self) -> char {
        match self.style {
            Style::Default => '›',
            Style::Compact => '▸',
            Style::Story => '◆',
        }
    }

    #[cfg(any(unix, windows))]
    pub fn feedback_ms(self) -> u64 {
        match self.style {
            Style::Default => 220,
            Style::Compact => 140,
            Style::Story => 320,
        }
    }

    #[cfg(any(unix, windows))]
    pub fn focused_stroke(self) -> f32 {
        match self.style {
            Style::Default => 0.24,
            Style::Compact => 0.20,
            Style::Story => 0.29,
        }
    }

    pub fn link(self, role: LinkInk, focused: bool) -> u8 {
        use LinkInk::*;
        match (self.background, role, focused) {
            (Background::Light, Focus, _) => 25,
            (Background::Light, FocusBright, _) => 27,
            (Background::Light, Structure | Unknown, false) => 60,
            (Background::Light, Structure | Unknown, true) => 53,
            (Background::Light, Match, false) => 29,
            (Background::Light, Match, _) => 22,
            (Background::Light, Confirmed, _) => 28,
            (Background::Light, Features, false) => 25,
            (Background::Light, Features, true) => 24,
            (Background::Light, Mismatch, false) => 160,
            (Background::Light, Mismatch, true) => 124,
            (Background::Light, Draft, _) => 94,
            (Background::Dark, Focus, _) => 141,
            (Background::Dark, FocusBright, _) => 177,
            (Background::Dark, Structure | Unknown, false) => 109,
            (Background::Dark, Structure | Unknown, true) => 153,
            (Background::Dark, Match, false) => 79,
            (Background::Dark, Match, _) => 122,
            (Background::Dark, Confirmed, _) => 157,
            (Background::Dark, Features, false) => 110,
            (Background::Dark, Features, true) => 117,
            (Background::Dark, Mismatch, false) => 209,
            (Background::Dark, Mismatch, true) => 216,
            (Background::Dark, Draft, _) => 222,
        }
    }

    pub fn face(self, shade: usize, selected: bool) -> u8 {
        let colors = match (self.background, self.style, selected) {
            (Background::Dark, Style::Default, true) => [22, 29, 36, 115],
            (Background::Dark, Style::Default, false) => [24, 31, 67, 110],
            (Background::Dark, Style::Compact, true) => [24, 30, 37, 123],
            (Background::Dark, Style::Compact, false) => [235, 239, 244, 249],
            (Background::Dark, Style::Story, true) => [54, 91, 98, 183],
            (Background::Dark, Style::Story, false) => [60, 67, 103, 146],
            (Background::Light, Style::Default, true) => [22, 28, 29, 72],
            (Background::Light, Style::Default, false) => [24, 25, 60, 67],
            (Background::Light, Style::Compact, true) => [23, 24, 30, 36],
            (Background::Light, Style::Compact, false) => [235, 238, 241, 244],
            (Background::Light, Style::Story, true) => [53, 54, 90, 97],
            (Background::Light, Style::Story, false) => [54, 60, 61, 97],
        };
        colors[shade]
    }
}

#[cfg(any(unix, windows))]
pub(super) fn xterm_rgb(index: u8) -> [u8; 3] {
    let shades = [0, 95, 135, 175, 215, 255];
    let index = index - 16;
    [
        shades[usize::from(index / 36)],
        shades[usize::from(index / 6 % 6)],
        shades[usize::from(index % 6)],
    ]
}

#[derive(Clone, Copy)]
pub(super) struct Material {
    #[cfg_attr(not(unix), allow(dead_code))]
    pub rgb: [u8; 3],
    pub grain_spacing: f64,
    pub grain_density: u8,
    #[cfg_attr(not(unix), allow(dead_code))]
    pub grain_size: f64,
}

#[derive(Clone, Copy)]
pub(super) struct SliceStyle {
    pub count: usize,
    pub fill: f64,
    pub footprint: [f64; 2],
    pub material: Material,
}

impl SliceStyle {
    pub const DEFAULT: Self = Self {
        count: 3,
        fill: 0.28,
        footprint: [1.0, 0.80],
        material: Material {
            rgb: [115, 157, 184],
            grain_spacing: 0.60,
            grain_density: 2,
            grain_size: 0.075,
        },
    };

    pub fn for_module(kind: &str) -> Self {
        let (count, fill, footprint, rgb, grain_spacing, grain_density) =
            match kind.rsplit('.').next().unwrap_or(kind) {
                // The point grain remains visible in Braille mode. Distinct
                // densities let common module families keep a visual signature
                // when terminal color is unavailable or disabled.
                "Embedding" | "EmbeddingBag" => (4, 0.26, [0.90, 0.75], [113, 191, 161], 0.44, 4),
                "Attention" | "MultiheadAttention" => {
                    (3, 0.24, [1.04, 0.87], [154, 157, 212], 0.78, 1)
                }
                "Linear" | "Bilinear" => (3, 0.26, [1.00, 0.76], [111, 177, 199], 0.58, 3),
                "Add" | "Add + Linear" => (3, 0.30, [0.81, 0.70], [193, 159, 126], 0.98, 2),
                "TransformerBlock" | "TransformerEncoderLayer" => {
                    (4, 0.25, [0.98, 0.71], [127, 159, 204], 0.38, 5)
                }
                _ => return Self::DEFAULT,
            };
        Self {
            count,
            fill,
            footprint,
            material: Material {
                rgb,
                grain_spacing,
                grain_density,
                ..Self::DEFAULT.material
            },
        }
    }

    pub fn texture_cue(kind: &str, locale: Locale) -> Option<&'static str> {
        match kind.rsplit('.').next().unwrap_or(kind) {
            "Embedding" | "EmbeddingBag" => {
                Some(locale.text("Dense Braille grain", "密集盲文颗粒", "密集盲文顆粒"))
            }
            "Attention" | "MultiheadAttention" => {
                Some(locale.text("Sparse Braille grain", "稀疏盲文颗粒", "稀疏盲文顆粒"))
            }
            "Linear" | "Bilinear" => {
                Some(locale.text("Medium Braille grain", "中等盲文颗粒", "中等盲文顆粒"))
            }
            "Add" | "Add + Linear" => {
                Some(locale.text("Open Braille grain", "疏朗盲文颗粒", "疏朗盲文顆粒"))
            }
            "TransformerBlock" | "TransformerEncoderLayer" => Some(locale.text(
                "Very dense Braille grain",
                "高密度盲文颗粒",
                "高密度盲文顆粒",
            )),
            _ => None,
        }
    }

    pub fn styled(mut self, style: Style) -> Self {
        match style {
            Style::Default => {}
            Style::Compact => {
                self.material.grain_spacing *= 2.0;
                self.material.grain_size *= 0.65;
            }
            Style::Story => {
                self.material.grain_spacing *= 0.8;
                self.material.rgb = self.material.rgb.map(|value| value.saturating_add(22));
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_texture_cues_match_distinct_densities_in_all_locales() {
        let modules = [
            (
                "Embedding",
                "Dense Braille grain",
                "密集盲文颗粒",
                "密集盲文顆粒",
            ),
            (
                "Attention",
                "Sparse Braille grain",
                "稀疏盲文颗粒",
                "稀疏盲文顆粒",
            ),
            (
                "Linear",
                "Medium Braille grain",
                "中等盲文颗粒",
                "中等盲文顆粒",
            ),
            ("Add", "Open Braille grain", "疏朗盲文颗粒", "疏朗盲文顆粒"),
            (
                "TransformerBlock",
                "Very dense Braille grain",
                "高密度盲文颗粒",
                "高密度盲文顆粒",
            ),
        ];
        let densities: std::collections::BTreeSet<_> = modules
            .iter()
            .map(|(kind, ..)| SliceStyle::for_module(kind).material.grain_density)
            .collect();
        assert_eq!(densities.len(), modules.len());

        for (kind, english, simplified, traditional) in modules {
            for locale in [Locale::English, Locale::Simplified, Locale::Traditional] {
                let expected = locale.text(english, simplified, traditional);
                assert_eq!(SliceStyle::texture_cue(kind, locale), Some(expected));
            }
        }
        assert_eq!(SliceStyle::texture_cue("ReLU", Locale::English), None);
    }
}
