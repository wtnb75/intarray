mod core;
mod error;
mod farray;
mod intarray;
mod rarray;

pub use error::ArrayError;
pub use farray::{FloatArray, FloatIter, BFLOAT16, FLOAT16, FLOAT32, FLOAT64};
pub use intarray::{IntArray, IntIter};
pub use rarray::{RadixArray, RadixIter};
