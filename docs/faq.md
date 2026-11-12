# FAQ

Back to the [README](../README.md).

## Is this a real Leica lens?

No. The PSFs are simulated in CODE V for a generic ~100 mm f/2 double-Gauss design (the CODE V sample design `ag_dblgauss`). It stands in for a Summicron-style double-Gauss and is not a measurement of any Leica lens.

Summicron and Leica are trademarks of Leica Camera AG. iSummicron is not affiliated with, sponsored by, or endorsed by Leica Camera AG.

## Why is it called iSummicron?

It aims at Summicron-style rendering for iPhone photos. The name is a nod to that aim, not a claim of affiliation.

## Why does my render look darker than the input?

The HDR tonemap takes its headroom from the brightest pixel after rendering. Blooming highlights raise that maximum, so the whole frame is tonemapped darker. On the sample photo the mean brightness is about 13 % lower than the input's. Tone fidelity is not solved in this version.

## Where are the swirl and cat-eye shapes?

They are the goal, not a finished feature. The shipped tensor varies with 12 radial bins rotated into 8 angular wedges, and highlights currently render as near-round discs. The tensor has no vignetting (pupil-clipping) term, so mechanical cat-eye is not modelled. Field effects are subtle in this version.

## Does it run on Linux or Windows?

Untested. The HDR path (`--hdr-source macos`) is macOS-only. Development and testing are on macOS 26 and 27 with Apple Silicon.

## Can I use my own PSFs?

Yes: pass `--tensor my_tensor.npz`. The required file layout is in the README section "Bring your own PSF tensor" ([README](../README.md#bring-your-own-psf-tensor)).

## Why is it not on PyPI?

Releases only, for now. Install the wheel from the GitHub Release (see the [README](../README.md#install)).

## Is the output deterministic?

Yes, for a given version, machine and set of options.

## What changes between versions?

See the [CHANGELOG](../CHANGELOG.md) and the [Releases page](https://github.com/chuang44-tiff/iSummicron/releases). Renders are not guaranteed to be identical between versions.
