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
    /// Owned here rather than inside `VideoPreview` because the scrub bar sits
    /// *above* the toggle strip, and a bar in the picture layer would sit under.
    @State private var video = VideoPreviewModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            content
            VStack(spacing: 0) {
                topBar
                Spacer(minLength: 0)
                if case .video = model.stage { scrubBar }
                // The toggles are meaningless over a spinner or an error, so the
                // strip appears with the picture it controls.
                switch model.stage {
                case .still, .video: controls
                case .loading, .failed: EmptyView()
                }
            }
        }
        .preferredColorScheme(.dark)
        .task { await model.load(media) }
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

    /// Top left, matching the spec. Nothing else lives up here yet — the share
    /// button is PR10's, and a disabled one would only promise something this
    /// build cannot do.
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

    private var scrubBar: some View {
        ScrubBar(model: video)
            .padding(.horizontal, 16).padding(.vertical, 8)
            .background(.black.opacity(0.55))
    }

    private var controls: some View {
        HStack(alignment: .center, spacing: 10) {
            Spacer(minLength: 0)
            ToggleBar(options: $model.overlay)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background(.black.opacity(0.55))
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
