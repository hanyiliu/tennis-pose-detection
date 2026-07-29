//  ToggleBar.swift
//  The overlay switches. Lives in its own file because the Photo Preview View
//  (PR8) shows the same four controls over a still or a scrubbed video.

import SwiftUI

/// The four independently toggleable overlay layers. Class label and confidence
/// are separate switches on purpose — the product spec lists them as two.
struct OverlayOptions: Equatable, Sendable {
    var boundingBox = true
    var keypoints = true
    var label = true
    var confidence = true
}

/// Camera-app-style control strip: compact, dark, thumb-reachable.
struct ToggleBar: View {
    @Binding var options: OverlayOptions

    var body: some View {
        HStack(spacing: 8) {
            chip("Box", "rectangle.dashed", $options.boundingBox)
            chip("Points", "figure.tennis", $options.keypoints)
            chip("Class", "tag.fill", $options.label)
            chip("Conf", "percent", $options.confidence)
        }
    }

    private func chip(_ title: String, _ symbol: String, _ isOn: Binding<Bool>) -> some View {
        let on = isOn.wrappedValue
        return Button { isOn.wrappedValue.toggle() } label: {
            VStack(spacing: 3) {
                Image(systemName: symbol).font(.system(size: 15, weight: .semibold))
                Text(title).font(.system(size: 10, weight: .medium))
            }
            .frame(width: 54, height: 46)
            .background(on ? Color.yellow : Color.white.opacity(0.16),
                        in: RoundedRectangle(cornerRadius: 11, style: .continuous))
            .foregroundStyle(on ? Color.black : Color.white)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(title)
        .accessibilityValue(on ? "on" : "off")
        .accessibilityAddTraits(on ? [.isSelected] : [])
    }
}
