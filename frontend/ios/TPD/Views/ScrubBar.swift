//  ScrubBar.swift
//  The scrubbable progress bar the spec puts along the bottom of the Photo Preview
//  View, plus the transport button that has nowhere else to live.

import SwiftUI

/// The exact-seek policy stays in `VideoPreviewModel`, not in a gesture handler.
struct ScrubBar: View {
    let model: VideoPreviewModel

    var body: some View {
        HStack(spacing: 14) {
            Button { model.togglePlay() } label: {
                Image(systemName: model.isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 16, weight: .bold))
                    .frame(width: 34, height: 34)
                    .background(.white.opacity(0.16), in: Circle())
                    .foregroundStyle(Color.white)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(model.isPlaying ? "Pause" : "Play")
            track
        }
    }

    private var track: some View {
        GeometryReader { proxy in
            let width = max(proxy.size.width, 1)
            let fraction = model.duration > 0 ? model.time / model.duration : 0
            let filled = width * min(max(fraction.isFinite ? fraction : 0, 0), 1)
            ZStack(alignment: .leading) {
                Capsule().fill(.white.opacity(0.22)).frame(height: 4)
                Capsule().fill(Color.yellow).frame(width: filled, height: 4)
                Circle().fill(Color.white).frame(width: 15, height: 15)
                    .shadow(radius: 2).offset(x: filled - 7.5)
            }
            .frame(maxHeight: .infinity)
            // The whole strip is the target, and `minimumDistance: 0` makes a
            // tap anywhere on it jump there.
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { model.scrub(to: $0.location.x / width, hasEnded: false) }
                    .onEnded { model.scrub(to: $0.location.x / width, hasEnded: true) }
            )
        }
        .frame(height: 30).accessibilityLabel("Video position")
    }
}
