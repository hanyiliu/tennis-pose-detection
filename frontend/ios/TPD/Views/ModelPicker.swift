//  ModelPicker.swift
//  The Camera-app mode selector, for models: a strip over the preview, no screen to leave.

import SwiftUI

struct ModelPicker: View {
    let models: [TPDModelEntry]
    let selected: TPDModelEntry?
    let select: (TPDModelEntry) -> Void
    @Environment(\.dynamicTypeSize) private var typeSize

    /// The picker's ceiling: 50 pt of strip, the 6 pt gap, then 144 pt of notes, from which
    /// `LiveCameraView` reserves a constant. Low on purpose — the class caption is centred on this
    /// same screen, and an explanation resting over the result it explains is worth less than none.
    static let maxHeight: CGFloat = 200

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
            // These reach ten lines each at an accessibility size and a VStack short of room
            // shrinks them, which truncated the warning mid-sentence. Bounded and scrolled instead,
            // visibly: a hard cut reads as a sentence that stopped; under a fade and a chevron, no.
            if typeSize.isAccessibilitySize, !Self.notes(for: selected).isEmpty {
                // The inset is the fade's own height: at the end, the last line clears it.
                ScrollView(.vertical) { notes.padding(.bottom, 48) }
                    .frame(height: Self.maxHeight - 56)
                    .mask(LinearGradient(colors: [.black, .black, .black, .clear],
                                         startPoint: .top, endPoint: .bottom))
                    .overlay(alignment: .bottom) { Image(systemName: "chevron.up.chevron.down")
                        .font(.caption2).padding(6).background(.ultraThinMaterial, in: Circle()) }
            } else { notes }
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
