//  ResultCache.swift
//  Finished results, keyed by frame index. Not an optimisation: one pass costs
//  ~9.7 s here, so without it scrubbing back a frame means waiting ten seconds
//  for an answer the app already had.

/// Keyed by **frame index**, never raw seconds: a scrub reports one time and the
/// tick behind it another, both describing the same picture. Unquantized, every
/// re-visit is a miss and the cache never hits.
@MainActor
final class ResultCache {
    let frameRate: Double
    private let capacity: Int
    private var entries: [Int: TPDResult] = [:]
    private var order: [Int] = []   // least-recently-used first
    /// 600 frames — 20 s at 30 fps. Entries are `TPDResult`s, not frames (~0.5 KB
    /// each, `labels` shared copy-on-write): the bound is so a long clip scrubbed
    /// end to end cannot grow without limit, not because memory is tight.
    init(frameRate: Double, capacity: Int = 600) {
        self.frameRate = frameRate > 0 ? frameRate : 30
        self.capacity = max(capacity, 1)
    }
    /// Frame *k* is on screen for `[k/fps, (k+1)/fps)`, so flooring — not rounding
    /// — maps a time to the frame a viewer sees. Non-finite and negative collapse
    /// to 0: `CMTime.seconds` is NaN before the item is ready and `Int(nan)` traps.
    func index(for seconds: Double) -> Int { Self.index(for: seconds, frameRate: frameRate) }

    /// The same arithmetic, reachable off this actor: `VideoExporter`'s analyze closure is
    /// neither async nor isolated, so an export reusing these entries keys them from its own
    /// task — and keying them one hair differently would miss every entry without looking wrong.
    nonisolated static func index(for seconds: Double, frameRate: Double) -> Int {
        guard seconds.isFinite, seconds > 0 else { return 0 }
        return Int((seconds * (frameRate > 0 ? frameRate : 30)).rounded(.down))
    }
    var count: Int { entries.count }
    /// Everything analysed so far, as a value. See `ResultSnapshot`.
    var snapshot: [Int: TPDResult] { entries }
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
    /// it returns**, so an abandoned pass leaves an honest miss rather than a hit
    /// serving an inference that never ended.
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
