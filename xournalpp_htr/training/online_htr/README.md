# OnlineHTR

Online (stroke-based) handwriting recognition: a bidirectional LSTM with CTC
loss after [Carbune et al. (2020)](https://doi.org/10.1007/s10032-020-00350-4),
ported from [OnlineHTR](https://github.com/PellelNitram/OnlineHTR). It uses
`(dx, dy, n)` features only, as Xournal++ stores no timestamps.

See the full documentation at [`docs/models/online_htr.md`](../../../docs/models/online_htr.md).
