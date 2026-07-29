//  LiveCameraView.swift
//  The home screen: live frames, overlays, and the controls that toggle them.

import SwiftUI

struct LiveCameraView: View {
    @State private var model = LiveViewModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            FramePreview(image: model.frame?.image)
            OverlayCanvas(result: model.frame?.result, options: model.overlay)
            if model.frame == nil { emptyState }
            VStack(spacing: 0) {
                Spacer(minLength: 0)
                controls
            }
        }
        .preferredColorScheme(.dark)
        // Cancelled on disappear; the loop's `defer` stops the source with it.
        .task { await model.run() }
    }

    private var controls: some View {
        HStack(alignment: .center, spacing: 10) {
            cameraRollButton
            Spacer(minLength: 0)
            ToggleBar(options: $model.overlay)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background(.black.opacity(0.55))
    }

    /// Bottom-left, exactly where the spec puts it — but inert. PR8 replaces the
    /// empty action with the PhotosPicker sheet and the Photo Preview View behind
    /// it; until then it is disabled rather than hidden, so the shell of the
    /// screen is the real one and nothing moves when the picker lands.
    private var cameraRollButton: some View {
        Button {} label: {
            VStack(spacing: 3) {
                Image(systemName: "photo.on.rectangle").font(.system(size: 15, weight: .semibold))
                Text("Library").font(.system(size: 10, weight: .medium))
            }
            .frame(width: 54, height: 46)
            .background(Color.white.opacity(0.16),
                        in: RoundedRectangle(cornerRadius: 11, style: .continuous))
            .foregroundStyle(Color.white)
        }
        .buttonStyle(.plain)
        .disabled(true)
        .opacity(0.45)
        .accessibilityLabel("Camera roll")
        .accessibilityHint("Not available yet")
    }

    /// Shown whenever there is no frame to draw — a fresh clone has no bundled
    /// clip, and the Simulator has no camera, so this is the *expected* first-run
    /// state, not an error page. It says what to do about it.
    private var emptyState: some View {
        VStack(spacing: 14) {
            Image(systemName: model.failure?.symbol ?? "camera.metering.center.weighted")
                .font(.system(size: 42, weight: .light))
                .foregroundStyle(.yellow)
            Text(model.failure?.title ?? "Starting the live feed…")
                .font(.headline)
                .multilineTextAlignment(.center)
            if let detail = model.failure?.detail {
                Text(detail)
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white.opacity(0.65))
            }
        }
        .padding(28)
        .frame(maxWidth: 360)
        .foregroundStyle(.white)
    }
}

#Preview {
    LiveCameraView()
}
