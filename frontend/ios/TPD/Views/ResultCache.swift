//  ResultCache.swift
//  Finished results, keyed by frame index. Not an optimisation: one pass costs
//  ~9.7 s here, so without it scrubbing back a frame means waiting ten seconds for
//  an answer the app already had.

/// Keyed by **frame index**, never raw seconds: a scrub reports one time and the
/// tick behind it another, both describing the same picture. Unquantized, every
/// re-visit is a miss and the cache never hits.
@MainActor
final class ResultCache {
    let frameRate: Double
    private let capacity: Int
    private var entries: [Int: TPDResult] = [:]
    private var order: [Int] = []   // least-recently-used first

    /// `capacity` defaults to 600 frames — 20 s at 30 fps. Entries are `TPDResult`s,
    /// not frames: a size, a rect, ~17 keypoints and a few floats (`labels` is
    /// copy-on-write, shared by every entry), so ~0.5 KB each and under 300 KB full.
    /// The bound is here so a long clip scrubbed end to end cannot grow without
    /// limit for a whole presentation, not because the memory is tight. A track with
    /// no nominal rate reports 0, which would make one bucket of the whole clip.
    init(frameRate: Double, capacity: Int = 600) {
        self.frameRate = frameRate > 0 ? frameRate : 30
        self.capacity = max(capacity, 1)
    }

    /// Frame *k* is on screen for `[k/fps, (k+1)/fps)`, so flooring — not rounding —
    /// maps a time to the frame a viewer is looking at. Non-finite and negative
    /// collapse to 0: `CMTime.seconds` is NaN for the invalid times `AVPlayer` hands
    /// out before its item is ready, and `Int(nan)` traps.
    func index(for seconds: Double) -> Int {
        guard seconds.isFinite, seconds > 0 else { return 0 }
        return Int((seconds * frameRate).rounded(.down))
    }

    var count: Int { entries.count }

    func value(at index: Int) -> TPDResult? {
        guard let hit = entries[index] else { return nil }
        touch(index)
        return hit
    }

    func insert(_ result: TPDResult, at index: Int) {
        // Overwriting a key changes no count, so only a new entry passes the bound.
        if entries.updateValue(result, forKey: index) == nil, entries.count > capacity,
           let victim = order.first {
            order.removeFirst()
            entries[victim] = nil
        }
        touch(index)
    }

    /// Read-through, and the contract worth naming: `analyse` is recorded **only if
    /// it returns**. A pass abandoned halfway throws, and this lets the error out
    /// without touching the store, so a later visit is an honest miss rather than a
    /// hit serving an inference that never finished.
    func result(at index: Int, analyse: () async throws -> TPDResult) async rethrows -> TPDResult {
        if let hit = value(at: index) { return hit }
        let fresh = try await analyse()
        insert(fresh, at: index)
        return fresh
    }

    private func touch(_ index: Int) {
        if let position = order.firstIndex(of: index) { order.remove(at: position) }
        order.append(index)
    }
}
