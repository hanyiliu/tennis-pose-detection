//  MediaPicker.swift
//  The Photo Selection View — the system picker, plus the two decisions that
//  belong to *selection* rather than to preview: telling a still from a movie,
//  and turning a still's file bytes into an upright image.

import CoreImage
import ImageIO
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// One selection, classified but not yet loaded. `id` is minted per pick, so
/// choosing the same asset twice re-presents the preview instead of being
/// swallowed as "no change" by `fullScreenCover(item:)`.
struct PickedMedia: Identifiable {
    enum Kind { case image, video }

    let id = UUID()
    let kind: Kind
    let item: PhotosPickerItem

    /// `supportedContentTypes` is the only thing the picker can answer *before*
    /// any bytes are transferred, which is exactly when the caller needs to know
    /// which preview to build. A movie advertises a `.movie`-conforming type
    /// (`.quickTimeMovie`, `.mpeg4Movie`); anything else is treated as a still,
    /// so an unfamiliar image UTI degrades into "try to decode it" rather than
    /// into the video placeholder.
    init(_ item: PhotosPickerItem) {
        self.item = item
        kind = item.supportedContentTypes.contains { $0.conforms(to: .movie) } ? .video : .image
    }
}

extension View {
    /// Presents the system photo picker as a sheet and reports each pick once.
    func mediaPicker(isPresented: Binding<Bool>,
                     onPick: @escaping (PickedMedia) -> Void) -> some View {
        modifier(MediaPickerModifier(isPresented: isPresented, onPick: onPick))
    }
}

private struct MediaPickerModifier: ViewModifier {
    @Binding var isPresented: Bool
    let onPick: (PickedMedia) -> Void
    @State private var selection: PhotosPickerItem?

    func body(content: Content) -> some View {
        content
            // **No `photoLibrary:` argument, deliberately.** Omitting it is what
            // keeps PHPicker running out of process: the app never sees the
            // library, so it needs no NSPhotoLibraryUsageDescription and the user
            // is never asked for read access. Passing `.shared()` would move the
            // picker in-process and cost both. Info.plist carries the *add* key
            // only, and that is the whole reason.
            //
            // `.current` rather than the default `.automatic`: the app wants the
            // asset's own bytes, EXIF and all. A transcode is free to bake the
            // rotation in and rewrite the orientation tag to 1, which would make
            // the orientation handling below untestable — and, on a device that
            // did not transcode, silently wrong.
            .photosPicker(isPresented: $isPresented, selection: $selection,
                          matching: .any(of: [.images, .videos]),
                          preferredItemEncoding: .current)
            .onChange(of: selection) { _, new in
                guard let new else { return }
                // Clearing it re-arms the binding; the resulting nil pass through
                // here is caught by the guard above.
                selection = nil
                onPick(PickedMedia(new))
            }
    }
}

/// The still half of a pick: file bytes -> a `CIImage` the engine can measure.
enum PickedImage {
    /// **EXIF orientation is applied here, and only here.**
    ///
    /// This is not cosmetic. `CGImageSourceCreateImageAtIndex` hands back the
    /// pixels as *stored*, which for anything shot in portrait is a landscape
    /// buffer plus an orientation tag the decoder did not act on. Feed that to
    /// the pipeline and stage 1 squashes a sideways picture into its 256x256
    /// square, so the bbox comes back describing a subject that is 90° from
    /// where the user sees it — a wrong answer that looks like a mapping bug and
    /// is really a decode bug. Rotating first makes the buffer, the preview and
    /// every overlay coordinate agree on one orientation.
    ///
    /// Missing tag -> 1 (`.up`), which is the TIFF default and a no-op transform.
    static func upright(from data: Data) throws -> CIImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw MediaPickerError.undecodableImage
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = properties?[kCGImagePropertyOrientation] as? Int32 ?? 1
        return CIImage(cgImage: decoded).oriented(forExifOrientation: orientation)
    }
}

/// The two ways a pick can fail before inference is even reached.
enum MediaPickerError: Error, Equatable, LocalizedError {
    /// The picker returned nothing — the asset is in iCloud and could not be
    /// fetched, or the transfer was cancelled.
    case transferFailed
    /// Bytes arrived but ImageIO could not make an image of them.
    case undecodableImage

    var errorDescription: String? {
        switch self {
        case .transferFailed:
            return "This item could not be loaded from your library. If it lives in iCloud, "
                + "open it once in Photos so the full-size copy is downloaded, then try again."
        case .undecodableImage:
            return "This file is not an image this device can decode."
        }
    }
}
