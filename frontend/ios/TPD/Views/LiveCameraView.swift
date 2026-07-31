//  LiveCameraView.swift
//  The home screen: live frames, overlays, and the controls that toggle them.

import SwiftUI

struct LiveCameraView: View {
    /// `.task(id:)`'s key. Bumping `attempt` restarts the feed; covering the
    /// screen flips `paused`, which cancels `run()` — the loop's `defer` then
    /// stops the source. Dismissing flips it back and a fresh run starts,
    /// re-using models the worker has already loaded.
    ///
    /// **Both** the picker and the preview count as covered. Either one hides
    /// the preview completely, and a capture session plus a three-stage pass per
    /// frame is not something to keep paying for while nothing it produces can
    /// be seen — least of all under PHPicker, which is a remote view hosted from
    /// another process and gets none of that work's benefit.
    private struct RunKey: Equatable { let attempt: Int, paused: Bool }

    @Environment(\.scenePhase) private var scenePhase
    @State private var model = LiveViewModel()
    @State private var isPicking = false
    @State private var picked: PickedMedia?

    private var runKey: RunKey {
        RunKey(attempt: model.attempt, paused: isPicking || picked != nil)
    }

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
        // Top-leading and inside the safe area: the bottom belongs to the
        // controls, and the empty state is centred, so nothing here is covered.
        .overlay(alignment: .topLeading) {
            DiagnosticsHUD(stats: model.performance, computeUnits: model.computeUnits)
                .padding(.horizontal, 14)
                .padding(.top, 6)
        }
        .preferredColorScheme(.dark)
        // Cancelled on disappear; the loop's `defer` stops the source with it.
        // Keyed on `runKey`, whose `attempt` half is the recovery token: `run()`
        // returns for good when the start fails, so re-entering the loop means
        // starting a *new* task, not resuming one.
        .task(id: runKey) { if !runKey.paused { await model.run() } }
        .mediaPicker(isPresented: $isPicking) { picked = $0 }
        .fullScreenCover(item: $picked) { media in
            PhotoPreviewView(media: media) { picked = nil }
        }
        // The other half of the same promise. The camera-denied state sends the
        // user to Settings; granting access there has to be enough on its own,
        // without them noticing a button back here.
        .onChange(of: scenePhase) { _, phase in
            if phase == .active, model.failure != nil { model.retry() }
        }
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

    /// Bottom-left, exactly where the spec puts it. Opens the system picker; the
    /// selection it reports is presented as `PhotoPreviewView` over this screen.
    private var cameraRollButton: some View {
        Button { isPicking = true } label: {
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
        .accessibilityLabel("Camera roll")
        .accessibilityHint("Choose a photo or video from your library")
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
                // Only once something has actually failed: while the first
                // attempt is still starting there is nothing to retry.
                Button("Try again") { model.retry() }
                    .font(.system(size: 15, weight: .semibold))
                    .buttonStyle(.borderedProminent)
                    .tint(.yellow)
                    .foregroundStyle(.black)
                    .accessibilityHint("Starts the live feed again")
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
