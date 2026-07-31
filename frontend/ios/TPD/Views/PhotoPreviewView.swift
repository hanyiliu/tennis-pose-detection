//  PhotoPreviewView.swift
//  The Photo Preview View: one picked still or clip, its overlays, and the same
//  four toggles the Live Camera View has.

import SwiftUI

/// Presented full-screen over the live view. It reuses `FramePreview`,
/// `OverlayCanvas` and `ToggleBar` **unchanged** — all three already take exactly
/// what this screen has (a `CGImage`, a `TPDResult`, a binding), because
/// `LiveViewModel` publishes the picture and its result as one value and this
/// view model publishes the same shape. There is no fork of the drawing code and
/// no second copy of the frame-to-view transform.
struct PhotoPreviewView: View {
    let media: PickedMedia
    /// Owned by the presenter, so exiting here and exiting by any other route
    /// (a swipe, a future share sheet) all run through one place.
    let onExit: () -> Void

    /// Fresh per presentation, which is what makes `.task` a once-per-pick load
    /// and what lets the models be released when the screen goes away.
    @State private var model = PreviewViewModel()
    /// Owned here, not in `VideoPreview`: the bar sits *above* the toggle strip.
    @State private var video = VideoPreviewModel()
    /// One per presentation, and it holds the export task. Nil until Share is
    /// tapped, so a preview nobody shares from costs nothing.
    @State private var export: ExportViewModel?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            content
            VStack(spacing: 0) {
                topBar
                Spacer(minLength: 0)
                if case .video = model.stage {
                    ScrubBar(model: video).padding(.horizontal, 16).padding(.vertical, 8)
                        .background(.black.opacity(0.55))
                }
                // The toggles are meaningless over a spinner or an error, so the
                // strip appears with the picture it controls.
                switch model.stage {
                case .still, .video: controls
                case .loading, .failed: EmptyView()
                }
            }
            // A sibling layer, not a screen of its own: `VideoPreview` never disappears behind it,
            // so the sweep it owns is stood down and picked back up from here.
            if let export {
                ExportProgressView(model: export) { self.export = nil; video.preload.resume() }
            }
        }
        .preferredColorScheme(.dark)
        .task { await model.load(media) }
        // Leaving mid-export must stop it: the run is minutes long and nothing it
        // produces from here on can be seen.
        .onDisappear { export?.stop() }
    }

    @ViewBuilder
    private var content: some View {
        switch model.stage {
        case .loading:
            status(symbol: nil, title: "Preparing…", detail: "Copying it out of your library.")
        case .still(let frame):
            FramePreview(image: frame.image)
            OverlayCanvas(result: frame.result, options: model.overlay)
        case .video(let url):
            VideoPreview(model: video, url: url, options: model.overlay)
        case .failed(let message):
            status(symbol: "exclamationmark.triangle", title: "Could not open this item",
                   detail: message)
        }
    }

    /// Top left, matching the spec. Nothing else lives up here: the share button
    /// belongs to the bottom-left slot beside the toggles.
    private var topBar: some View {
        HStack {
            Button(action: onExit) {
                Image(systemName: "xmark")
                    .font(.system(size: 15, weight: .bold))
                    .frame(width: 38, height: 38)
                    .background(.black.opacity(0.55), in: Circle())
                    .foregroundStyle(Color.white)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Close")
            .accessibilityHint("Returns to the live camera view")
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.top, 8)
    }

    private var controls: some View {
        HStack(alignment: .center, spacing: 10) {
            shareButton
            Spacer(minLength: 0)
            ToggleBar(options: $model.overlay)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background(.black.opacity(0.55))
    }

    /// Bottom-left, the slot the camera roll button occupies on the live view. Live
    /// only over a movie — a still has no clip to burn an overlay into — and dimmed
    /// rather than hidden, so the control does not vanish between two picks. The
    /// toggles are read at the tap, so what is burned in is what is on screen and
    /// changing them mid-export cannot rewrite it.
    private var shareButton: some View {
        let url: URL? = if case .video(let url) = model.stage { url } else { nil }
        return Button {
            guard let url else { return }
            // Before the snapshot, so it carries off everything the sweep produced.
            video.preload.suspend()
            let started = ExportViewModel()
            started.start(source: url, overlay: model.overlay, snapshot: video.analysed)
            export = started
        } label: {
            VStack(spacing: 3) {
                Image(systemName: "square.and.arrow.up").font(.system(size: 15, weight: .semibold))
                Text("Share").font(.system(size: 10, weight: .medium))
            }
            .frame(width: 54, height: 46)
            .background(Color.white.opacity(0.16),
                        in: RoundedRectangle(cornerRadius: 11, style: .continuous))
            .foregroundStyle(Color.white)
            .opacity(url == nil ? 0.35 : 1)
        }
        .buttonStyle(.plain)
        .disabled(url == nil)
        .accessibilityLabel("Share")
        .accessibilityHint(url == nil ? "Only videos can be exported"
                                      : "Burns the overlay in and saves the video to your camera roll")
    }

    /// Spinner, placeholder and error are the same layout with different
    /// contents; a nil symbol means "still working", which is the only one of the
    /// three that gets a spinner instead of a glyph.
    private func status(symbol: String?, title: String, detail: String) -> some View {
        VStack(spacing: 14) {
            if let symbol {
                Image(systemName: symbol)
                    .font(.system(size: 42, weight: .light))
                    .foregroundStyle(.yellow)
            } else {
                ProgressView()
                    .controlSize(.large)
                    .tint(.yellow)
            }
            Text(title).font(.headline).multilineTextAlignment(.center)
            Text(detail)
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.65))
        }
        .padding(28)
        .frame(maxWidth: 360)
        .foregroundStyle(.white)
        .accessibilityElement(children: .combine)
    }
}
