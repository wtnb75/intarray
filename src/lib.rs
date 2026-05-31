mod core;
mod error;
mod intarray;
mod rarray;

pub use error::ArrayError;
pub use intarray::{IntArray, IntIter};
pub use rarray::{RadixArray, RadixIter};
