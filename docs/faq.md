# FAQ

Back to the [README](../README.md).

## Is this a real Leica lens?

No. The blur shapes (PSFs) are ray-traced from two public lens prescriptions, not measured from any lens:

- **35 mm f/2 (the default):** ray-traced from the expired US patent 3,006,249, Example II, using optiland.
- **50 mm f/2 (`--lens 50`):** the double-Gauss sample prescription that ships with optiland (MIT license).

The patent gives curvatures and spacings but not the size of each piece of glass, so the rim sizes follow a stated rule. They are an assumption.

Summicron and Leica are trademarks of Leica Camera AG. iSummicron is not affiliated with, sponsored by, or endorsed by Leica Camera AG.

## Why is it called iSummicron?

It aims at Summicron-style rendering for iPhone photos. The name is a nod to that aim, not a claim of affiliation.

## Why does my render look darker than the input?

With the default look the final tone follows the input photo's own SDR tone (`--tonemap input-lut`), so the render should be close to the input's brightness. If you pass `--tonemap ci` (the Apple Core Image tonemap), the tonemap takes its headroom from the brightest pixel after rendering; blooming highlights raise that maximum and the whole frame can come out darker. Renders from before 0.4.0 used that tonemap.

## Where are the swirl and cat-eye shapes?

They are traced into the shipped blur shapes. Toward the frame edge the rims of the glass clip the cone of light, which gives the cat-eye, and the renderer turns each shape to face away from the frame center, which gives the swirl. Look at out-of-focus point lights near the edges of the frame; `--lens 50` shows the swirl most, and the default 35 mm shows hard clipping in the corners.

Two things are not modelled: the corners are not darkened (each shape keeps its full brightness), and the shapes turn in 8 steps of 45° around the frame. See [How it works](how-it-works.md#limits).

## Which lens should I pick?

The default 35 mm f/2 gives smaller blur and clips hard in the corners. `--lens 50` gives larger blur and stronger swirl toward the edges. The lens changes the blur shapes only; it does not crop or reframe your photo.

## What happened to the older kernel options?

`--kernel-family`, `--lens-aperture-mm` and `--psf-norm` were removed in 0.5.0, and the bundled PSF table was replaced (kernels are always L1-normalized now). See "Migrating to 0.5.0" in the [CHANGELOG](../CHANGELOG.md).

## Does it run on Linux or Windows?

Untested. The HDR path (`--hdr-source macos`) is macOS-only. Development and testing are on macOS 26 and 27 with Apple Silicon.

## Can I use my own PSFs?

Yes: pass `--tensor my_tensor.npz` (not together with `--lens`). The required file layout is in the usage guide under [Bring your own PSF tensor](usage.md#bring-your-own-psf-tensor).

## Why is it not on PyPI?

Releases only, for now. Install the wheel from the GitHub Release (see the [README](../README.md#install)).

## Is the output deterministic?

Yes, for a given version, machine and set of options.

## What changes between versions?

See the [CHANGELOG](../CHANGELOG.md) and the [Releases page](https://github.com/chuang44-tiff/iSummicron/releases). Renders are not guaranteed to be identical between versions.
