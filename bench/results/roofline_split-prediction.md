# Prediction, written before the measurement ran

Scored by `rust/tools/roofline_split.py`. Written first so the artifact can contradict it.

Standing figure that must not be reused as the answer: `executed_by = {'CPUExecutionProvider': 120,
'VulkanExecutionProvider': 99}`. That is a node count. The question is what fraction of the *work*
those 120 nodes are.

1. **The CPU share of bytes rises steeply with context and the CPU share of FLOPs rises with it.**
   The 32 declined `GroupQueryAttention` nodes are the only nodes in the graph whose cost depends on
   context at all; every `MatMulNBits` is context-independent at a decode step. So the curve should
   be near-flat in FLOPs at ctx=0 and climb monotonically.

2. **At ctx=0 the CPU share of bytes is small — under 5%.** At zero context there is no KV cache to
   read, so attention moves almost nothing. This is why the existing quoted figure hides the cost.

3. **At ctx=8192 the CPU share of bytes is large — over 40%.** KV traffic was measured at 60.5% of
   bytes at 8192, and KV traffic is entirely inside the declined nodes.

4. **The node count will not predict either number.** 120/366 = 32.8% of nodes; I expect the byte
   fraction to be far below that at ctx=0 and to cross it somewhere in the middle of the range.

5. **The EP's own estimator will report a CPU FLOP share close to the anchor-node ratio and will not
   move with context at all**, because anchors are scored with the constant `2*3072*3072` and the
   rest with `out_bytes/2` under a substituted dim. If it does move, I have misread `ep.rs`.

6. **Fabricated extents will contribute nothing**, because a stated context length resolves every
   dim in this graph. If that is wrong the answer is `UNOBSERVABLE`, not small.
