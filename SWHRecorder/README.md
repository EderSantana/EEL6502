Noise cancelletion demo.

Run: python realTimeAudio.py

This code is basically Scott W. Harden's code [1] with some mods for 440Hz tone cancellation.

You can generate the tone on the following web site: http://onlinetonegenerator.com

The noise cancellation is based on the Wiener filter. We store a 100 ms sine wave and it's order 2 autocorrelation function, thus, we have a order 2 Wiener Filter. Note that this is enough for sine waves due to its simple implicit dimmension. We do not make the experiment with random noise because MacBooks have a really nice adaptive filter for that kind of noise built-in.

[1] http://www.swharden.com/blog/2013-05-09-realtime-fft-audio-visualization-with-python/
