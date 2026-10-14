//! The method: Robinson's per-token probability and Fisher's chi-squared combining, as SpamBayes
//! does them.
//!
//! # Per token: Robinson's f(w)
//!
//! For a token `w` seen in `s` of `nspam` spam messages and `h` of `nham` ham messages, Graham's
//! ratio (*A Plan for Spam*) is
//!
//! ```text
//! p(w) = (s / nspam) / (s / nspam + h / nham)
//! ```
//!
//! and Robinson's correction shrinks it toward a prior `x` with strength `S`, by how much
//! evidence there is (`n = s + h`):
//!
//! ```text
//! f(w) = (S·x + n·p(w)) / (S + n)
//! ```
//!
//! A token never seen has `n = 0`, so `f(w) = x`. SpamBayes's defaults are `x = 0.5`,
//! `S = 0.45`, and so are these ([`Settings::default`]). As SpamBayes does, a message count of
//! zero is read as one, so a model trained only on spam still has defined ratios.
//!
//! # Per message: Fisher's combining
//!
//! Only the tokens whose `f(w)` is at least [`Settings::minimum_prob_strength`] away from 0.5
//! count (the *clues*), at most [`Settings::max_discriminators`] of them, strongest first. With
//! `n` clues,
//!
//! ```text
//! S = 1 − C⁻¹(−2·Σ ln(1 − f(w)), 2n)      (evidence of spam)
//! H = 1 − C⁻¹(−2·Σ ln f(w),       2n)      (evidence of ham)
//! score = (1 + S − H) / 2
//! ```
//!
//! where `C⁻¹(χ², v)` is the chi-squared survival function ([`chi2q`]). A message with no clues
//! scores exactly 0.5. Because `S` and `H` are judged separately, a message with strong evidence
//! BOTH ways scores near 0.5 instead of being forced to a side: that is the *unsure* band, and
//! why a cutoff pair ([`Settings::ham_cutoff`], [`Settings::spam_cutoff`]) classifies rather than
//! one threshold.
//!
//! ```
//! use ikigai_bayes::classify::{chi2q, fisher, robinson, Settings};
//!
//! let settings = Settings::default();
//! // One spam message containing the token, one ham message without it.
//! let f = robinson(1, 0, 1, 1, &settings);
//! assert!((f - 1.225 / 1.45).abs() < 1e-12);
//! // One clue: the score IS that clue.
//! assert!((fisher(&[f]) - f).abs() < 1e-12);
//! // The survival function at 2 degrees of freedom is exp(−χ²/2).
//! assert!((chi2q(2.0, 2) - (-1.0f64).exp()).abs() < 1e-15);
//! ```

/// The method's parameters, per model. Every default is SpamBayes's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Robinson's prior for a token never seen, `x`.
    pub unknown_word_prob: f64,
    /// Robinson's strength of that prior, `S`, in messages.
    pub unknown_word_strength: f64,
    /// How far from 0.5 a token's probability must be to count as a clue.
    pub minimum_prob_strength: f64,
    /// The most clues a score combines.
    pub max_discriminators: usize,
    /// A score below this is `ham`.
    pub ham_cutoff: f64,
    /// A score at or above this is `spam`; between the two cutoffs it is `unsure`.
    pub spam_cutoff: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            unknown_word_prob: 0.5,
            unknown_word_strength: 0.45,
            minimum_prob_strength: 0.1,
            max_discriminators: 150,
            ham_cutoff: 0.20,
            spam_cutoff: 0.90,
        }
    }
}

impl Settings {
    /// These settings with other cutoffs. Refused unless `0 <= ham <= spam <= 1`, both finite.
    pub fn with_cutoffs(mut self, ham: f64, spam: f64) -> Result<Self, String> {
        if !(ham.is_finite() && spam.is_finite() && 0.0 <= ham && ham <= spam && spam <= 1.0) {
            return Err(format!(
                "cutoffs must satisfy 0 <= ham <= spam <= 1 (got ham {ham}, spam {spam})"
            ));
        }
        self.ham_cutoff = ham;
        self.spam_cutoff = spam;
        Ok(self)
    }

    /// The class a score falls in under these cutoffs.
    pub fn class_of(&self, score: f64) -> Class {
        if score < self.ham_cutoff {
            Class::Ham
        } else if score >= self.spam_cutoff {
            Class::Spam
        } else {
            Class::Unsure
        }
    }
}

/// What a score says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Class {
    /// Below the ham cutoff.
    Ham,
    /// Between the cutoffs.
    Unsure,
    /// At or above the spam cutoff.
    Spam,
}

impl Class {
    /// `ham`, `unsure` or `spam`.
    pub fn as_str(self) -> &'static str {
        match self {
            Class::Ham => "ham",
            Class::Unsure => "unsure",
            Class::Spam => "spam",
        }
    }
}

/// Robinson's f(w) for a token in `spam` of `nspam` spam messages and `ham` of `nham` ham ones.
pub fn robinson(spam: u64, ham: u64, nspam: u64, nham: u64, settings: &Settings) -> f64 {
    let n = (spam + ham) as f64;
    if n == 0.0 {
        return settings.unknown_word_prob;
    }
    // SpamBayes: a zero message count is read as one.
    let spam_ratio = spam as f64 / nspam.max(1) as f64;
    let ham_ratio = ham as f64 / nham.max(1) as f64;
    let p = spam_ratio / (spam_ratio + ham_ratio);
    let s = settings.unknown_word_strength;
    (s * settings.unknown_word_prob + n * p) / (s + n)
}

/// The chi-squared survival function `Q(χ², v)` (the probability that a chi-squared variable
/// with `v` degrees of freedom exceeds `chi2`), for EVEN `v`, by the series SpamBayes uses:
/// `exp(−m) · Σ_{i<v/2} mⁱ/i!` with `m = χ²/2`, capped at 1.
///
/// ```
/// use ikigai_bayes::classify::chi2q;
///
/// // v = 4: exp(−m)·(1 + m); at χ² = 4, m = 2, so 3·exp(−2).
/// assert!((chi2q(4.0, 4) - 3.0 * (-2.0f64).exp()).abs() < 1e-15);
/// assert_eq!(chi2q(0.0, 6), 1.0);
/// ```
pub fn chi2q(chi2: f64, v: usize) -> f64 {
    debug_assert!(
        v.is_multiple_of(2),
        "chi2q takes an even number of degrees of freedom"
    );
    let m = chi2 / 2.0;
    let mut term = (-m).exp();
    let mut sum = term;
    for i in 1..v / 2 {
        term *= m / i as f64;
        sum += term;
    }
    sum.min(1.0)
}

/// Fisher's combining of clue probabilities into one score in `[0, 1]`; 0.5 for no clues.
///
/// Each probability must lie strictly between 0 and 1, which every Robinson f(w) does (the prior
/// keeps it off both ends).
pub fn fisher(clues: &[f64]) -> f64 {
    if clues.is_empty() {
        return 0.5;
    }
    let n = clues.len();
    let ln_not: f64 = clues.iter().map(|p| (1.0 - p).ln()).sum();
    let ln_is: f64 = clues.iter().map(|p| p.ln()).sum();
    let s = 1.0 - chi2q(-2.0 * ln_not, 2 * n);
    let h = 1.0 - chi2q(-2.0 * ln_is, 2 * n);
    (s - h + 1.0) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn robinson_hand_values() {
        let s = Settings::default();
        // Never seen: the prior.
        assert_eq!(robinson(0, 0, 10, 10, &s), 0.5);
        // In 1 of 1 spam and 0 of 1 ham: p = 1, n = 1 → (0.225 + 1) / 1.45.
        assert!(close(robinson(1, 0, 1, 1, &s), 1.225 / 1.45));
        // In 0 of 4 spam and 2 of 4 ham: p = 0, n = 2 → 0.225 / 2.45.
        assert!(close(robinson(0, 2, 4, 4, &s), 0.225 / 2.45));
        // In 3 of 10 spam and 1 of 5 ham: ratios 0.3 and 0.2, p = 0.6, n = 4
        // → (0.225 + 2.4) / 4.45.
        assert!(close(robinson(3, 1, 10, 5, &s), 2.625 / 4.45));
        // A zero message count reads as one: spam only, no ham messages at all.
        assert!(close(robinson(2, 0, 2, 0, &s), (0.225 + 2.0) / 2.45));
    }

    #[test]
    fn chi2q_hand_values() {
        assert!(close(chi2q(0.0, 2), 1.0));
        assert!(close(chi2q(2.0, 2), (-1.0f64).exp()));
        // v = 6: exp(−m)(1 + m + m²/2), m = 3.
        assert!(close(chi2q(6.0, 6), (-3.0f64).exp() * (1.0 + 3.0 + 4.5)));
        // Large χ² underflows to 0, never NaN.
        assert_eq!(chi2q(1e6, 4), 0.0);
    }

    #[test]
    fn fisher_hand_values() {
        assert_eq!(fisher(&[]), 0.5);
        // One clue: S = p, H = 1 − p, so the score is p.
        assert!(close(fisher(&[0.8]), 0.8));
        assert!(close(fisher(&[0.1]), 0.1));
        // Two clues 0.9 and 0.8, worked by hand:
        //   S = 1 − Q(−2 ln(0.1·0.2), 4); m = −ln 0.02; Q = 0.02·(1 + m)
        //   H = 1 − Q(−2 ln(0.9·0.8), 4); m = −ln 0.72; Q = 0.72·(1 + m)
        let ms = -(0.02f64).ln();
        let mh = -(0.72f64).ln();
        let s = 1.0 - 0.02 * (1.0 + ms);
        let h = 1.0 - 0.72 * (1.0 + mh);
        let want = (s - h + 1.0) / 2.0;
        assert!(close(fisher(&[0.9, 0.8]), want));
        assert!((want - 0.929_141).abs() < 1e-6, "{want}");
    }

    #[test]
    fn opposed_evidence_is_unsure() {
        // Strong evidence both ways lands in the middle, not on a side.
        let score = fisher(&[0.99, 0.99, 0.01, 0.01]);
        assert!(close(score, 0.5), "{score}");
        assert_eq!(Settings::default().class_of(score), Class::Unsure);
    }

    #[test]
    fn cutoffs_classify() {
        let s = Settings::default();
        assert_eq!(s.class_of(0.199), Class::Ham);
        assert_eq!(s.class_of(0.20), Class::Unsure);
        assert_eq!(s.class_of(0.899), Class::Unsure);
        assert_eq!(s.class_of(0.90), Class::Spam);
        let tight = s.with_cutoffs(0.5, 0.5).unwrap();
        assert_eq!(tight.class_of(0.4999), Class::Ham);
        assert_eq!(tight.class_of(0.5), Class::Spam);
        assert!(s.with_cutoffs(0.9, 0.2).is_err());
        assert!(s.with_cutoffs(-0.1, 0.5).is_err());
        assert!(s.with_cutoffs(0.1, f64::NAN).is_err());
    }
}
