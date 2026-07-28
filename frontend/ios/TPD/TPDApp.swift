import SwiftUI

@main
struct TPDApp: App {
    var body: some Scene {
        WindowGroup {
            ScaffoldRootView()
        }
    }
}

/// Placeholder root. Replaced by `LiveCameraView` once the inference core and
/// frame sources land; until then it exists so the target has a real entry
/// point that builds and launches.
struct ScaffoldRootView: View {
    var body: some View {
        VStack(spacing: 10) {
            Text("TPD")
                .font(.system(size: 44, weight: .semibold, design: .rounded))
            Text("Tennis Pose Detection")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Scaffold build — camera, Core ML inference and overlays are not wired up yet.")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(.tertiary)
                .padding(.top, 6)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
        .foregroundStyle(.white)
        .preferredColorScheme(.dark)
    }
}

#Preview {
    ScaffoldRootView()
}
