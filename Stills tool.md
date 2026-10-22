# Description

This tool extracts the best quality stills from videos. It is an automated tool that should work without having a cloud LLM working on it. However, an agent should be able to execute commands for this on its own without relying on the GUI.

The quality of the stills comes from this criteria:

- We need an overall representation of the whole video. At least one frame per second, then, you choose the best ones, and discard the bad ones. Finally, you should have:
	- 20 high-quality frames for 30+ second videos.
	- 15 high-quality frames for 30 second videos.
	- 10 high-quality frames for 15 second videos.
	- 6 high-quality frames for 6 second videos.
- Optionally, when asked, you send a report of the frames score. Low, mid, and high. The user  can manually select the best ones based on your criteria. However, the idea is to not have to do this manually most of the times, its just an option.
- You go through the whole video, identify a good frame, and take a still.
- A good frame has the least ammount of motion blur. This counts for text and people. I'm not sure how to do it.
- You delete duplicated frames. This can be done by doing a difference pass through all stills if the difference is 0 they're duplicated. If there's a better way please suggest it.
- The text should be well aligned and have 100% opacity. I'm not sure how to implement this. Maybe having a local AI model to identfy the text and then check that the full color opacity of a sample of adjacent frames corresponds to the full opacity for all the text? What's the best wa to do this?
- Great stills have people with open eyes and smiling faces. I'm thinking of a local AI model to detect these. If there's a better way please suggest it.
- The end-card is the last frame.

We already have a rudimentary stills extractor D:\Filmkraft\Tools\Extract_stills but it doesn't has this qualitative way to extract the best stills of a video.

There's a GUI similar to the one you find in D:\Filmkraft\Tools\Technical_QC

I has to be compatible with Windows and MacOS.

The stills are saved by default as PNG 8 bits, but the output supports TIFF, and JPEG as well. And 16 bit variations.

This is the criteria I think, do you have additional criteria we should follow? Do you have alternative ways to do it? Do you have any questions?