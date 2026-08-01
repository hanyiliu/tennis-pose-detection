//  ModelPicker.swift
//  The Camera-app mode selector, for models: a strip over the preview, no screen to leave.

import SwiftUI

/// Every model the registry lists, plus the notes the *selected* one owes the user.
struct ModelPicker: View {
    let models: [TPDModelEntry]
    let selected: TPDModelEntry?
    let select: (TPDModelEntry) -> Void
    @Environment(\.dynamicTypeSize) private var typeSize

    /// The picker's ceiling: 50 pt of strip, the 6 pt gap, then the notes. Chip point sizes are
    /// fixed, so only the notes move with Dynamic Type and `LiveCameraView` can reserve a constant.
    static let maxHeight: CGFloat = 356

    var body: some View {
        let notes = VStack(alignment: .leading, spacing: 6) {
            ForEach(Self.notes(for: selected), id: \.self) { note in
                Text(note).font(.caption2).foregroundStyle(.white.opacity(0.85))
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 15))
            }
        }
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) { ForEach(models) { chip($0) } }
                .padding(5).background(.ultraThinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.18)))
            // These reach ten lines each at an accessibility size, and a VStack short of room
            // shrinks them: the class-order warning truncated mid-sentence while the strip left the
            // top of the screen. Bounded and scrolled there, no chip moves and no word is lost.
            // Branched, not measured — `ViewThatFits` sized to the greedy scroll view either way.
            if typeSize.isAccessibilitySize, !Self.notes(for: selected).isEmpty {
                ScrollView(.vertical) { notes }.frame(height: Self.maxHeight - 56).scrollIndicators(.visible)
            } else {
                notes
            }
        }
        .foregroundStyle(.white)
    }

    private func chip(_ entry: TPDModelEntry) -> some View {
        let on = entry.id == selected?.id
        return Button { select(entry) } label: {
            VStack(spacing: 2) {
                Text(entry.shortName).font(.system(size: 12, weight: .semibold))
                Text("\(entry.input.size)px").font(.system(size: 10, weight: .medium))
                    .foregroundStyle(on ? .black.opacity(0.6) : .white.opacity(0.6))
            }
            .lineLimit(1).padding(.horizontal, 12).frame(height: 40)
            .background(on ? Color.yellow : Color.white.opacity(0.16), in: Capsule())
            .foregroundStyle(on ? Color.black : Color.white).contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(entry.displayName)  // the chip is abbreviated; VoiceOver is not
        .accessibilityAddTraits(on ? [.isSelected] : [])
    }

    static func notes(for entry: TPDModelEntry?) -> [String] {
        guard let entry else { return [] }
        return [
            entry.producesGeometry ? nil : "Classifier: one image in, \(entry.labels.count) "
                + "numbers out. No box or keypoints to draw, so those switches are off here.",
            entry.labelOrder.status != .assumed ? nil : "Class order unverified: these names are "
                + "assumed from the training code, with no dataset to confirm them — a prediction "
                + "may be reported under the wrong name.",
        ].compactMap { $0 }
    }
}
