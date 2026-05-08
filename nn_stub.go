//go:build !nn

package main

// Stubs used when building without the nn build tag (no ONNX dependency).

// ── Old single-input model stubs ──────────────────────────────────────────────

type nnCtx struct{}

func InitNN(modelPath, libPath string) error   { panic("NN not built in") }
func newNNCtx() (*nnCtx, error)               { panic("NN not built in") }
func (c *nnCtx) destroy()                     {}
func (c *nnCtx) eval(b *board) int            { panic("NN not built in") }
func playMCNNAllCands(b board, nc *nnCtx) int { panic("NN not built in") }

// ── New ResNet+aux model stubs ────────────────────────────────────────────────

type nnCtxV2 struct{}

func newNNCtxV2(modelPath string) (*nnCtxV2, error) { panic("NN not built in") }
func (c *nnCtxV2) destroy()                         {}
func (c *nnCtxV2) eval(b *board) int                { panic("NN not built in") }
func playModelAnA(b board, nc *nnCtxV2) int         { panic("NN not built in") }
