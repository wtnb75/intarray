use crate::error::ArrayError;

pub(crate) trait PackedArrayCore {
    type Item: Copy;

    fn raw_get(&self, i: usize) -> Result<Self::Item, ArrayError>;
    fn raw_set(&mut self, i: usize, v: Self::Item) -> Result<(), ArrayError>;
    fn raw_len(&self) -> usize;
    fn raw_resize(&mut self, len: usize);

    fn core_push(&mut self, v: Self::Item) -> Result<usize, ArrayError> {
        self.raw_resize(self.raw_len() + 1);
        let len = self.raw_len();
        match self.raw_set(len - 1, v) {
            Ok(()) => Ok(len - 1),
            Err(e) => {
                self.raw_resize(len - 1);
                Err(e)
            }
        }
    }

    fn core_pop(&mut self) -> Result<Self::Item, ArrayError> {
        let len = self.raw_len();
        if len == 0 {
            return Err(ArrayError::Empty);
        }
        let v = self.raw_get(len - 1)?;
        self.raw_resize(len - 1);
        Ok(v)
    }

    fn core_extend<I>(&mut self, vals: I) -> Result<(), ArrayError>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        let orig = self.raw_len();
        for v in vals {
            if let Err(e) = self.core_push(v) {
                self.raw_resize(orig);
                return Err(e);
            }
        }
        Ok(())
    }
}
