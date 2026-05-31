use std::fmt;

#[derive(Debug, PartialEq)]
pub enum ArrayError {
    OutOfBounds,
    TooLarge,
    TooSmall,
    Empty,
    InvalidRange,
}

impl fmt::Display for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds => write!(f, "index out of bounds"),
            Self::TooLarge => write!(f, "value too large"),
            Self::TooSmall => write!(f, "value too small"),
            Self::Empty => write!(f, "array is empty"),
            Self::InvalidRange => write!(f, "invalid range: K must be in 2..=u64::MAX"),
        }
    }
}

impl std::error::Error for ArrayError {}
