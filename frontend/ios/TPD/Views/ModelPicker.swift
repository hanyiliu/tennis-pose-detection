//  ModelPicker.swift
//  The Camera-app mode selector, for models: a strip over the preview, no screen to leave.

import SwiftUI

/// Every model the registry lists, plus the notes the *selected* one owes the user.
struct ModelPicker: View {
    let models: [TPDModelEntry]
    let selected: TPDModelEntry?
    let select: (TPDModelEntry) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) { ForEach(models) { chip($0) } }
                .padding(5).background(.ultraThinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.18)))
            ForEach(Self.notes(for: selected), id: \.self) { note in
                Text(note).font(.caption2).foregroundStyle(.white.opacity(0.85))
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .background(.ultraThinMaterial, in: Capsule())
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
                + "assumed from the training code, with no dataset to confirm them. A prediction "
                + "may be reported under the wrong name.",
        ].compactMap { $0 }
    }
}
