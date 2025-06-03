import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, Agent, AgentSession
from livekit.agents.llm import ChatContext, ChatMessage, ImageContent
from livekit.agents.llm.tool_context import ToolContext, function_tool
from livekit.plugins import deepgram, openai, silero

# Load environment variables from a .env file
load_dotenv()


class AssistantFunction(ToolContext):
    """This class is used to define functions that will be called by the assistant."""

    def __init__(self):
        # Initialize ToolContext with an empty list of tools initially
        super().__init__([])
        self.latest_image: rtc.VideoFrame | None = None

    @function_tool(
        description=(
            "Called when asked to evaluate something that would require vision capabilities, "
            "for example, an image, video, or the webcam feed. Call this when user asks to "
            "look at something, describe what they see, or analyze an image."
        )
    )
    async def analyze_image(
        self,
        user_msg: Annotated[
            str,
            "The user message that triggered this function",
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        if self.latest_image:
            return "I can see the current video feed. Let me analyze what's visible."
        else:
            return "I don't currently have access to any images to analyze."

    def update_image(self, image: rtc.VideoFrame):
        """Update the latest image for vision analysis."""
        self.latest_image = image


class VisionAssistant(Agent):
    """Main assistant class with vision capabilities."""
    
    def __init__(self):
        super().__init__(
            instructions=(
                "Your name is Alloy. You are a funny, witty bot with vision capabilities. "
                "Your interface with users will be voice and vision. "
                "Respond with short and concise answers. "
                "Avoid using unpronounceable punctuation or emojis. "
                "When users ask you to look at something or describe what you see, "
                "call the analyze_image function to use your vision capabilities."
            )
        )


async def get_video_track(room: rtc.Room):
    """Get the first video track from the room."""
    
    # Wait for participants to join
    for _ in range(30):  # Wait up to 30 seconds
        for _, participant in room.remote_participants.items():
            for _, track_publication in participant.track_publications.items():
                if track_publication.track is not None and isinstance(
                    track_publication.track, rtc.RemoteVideoTrack
                ):
                    print(f"Using video track {track_publication.track.sid}")
                    return track_publication.track
        await asyncio.sleep(1)
    
    return None


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    # Initialize the assistant and function context
    assistant = VisionAssistant()
    function_ctx = AssistantFunction()
    
    # Create the agent session with all components
    session = AgentSession(
        vad=silero.VAD.load(),  # Voice Activity Detector
        stt=deepgram.STT(),     # Speech To Text
        llm=openai.LLM(model="gpt-4o"),  # Language Model with vision support and tools
        tts=openai.TTS(voice="alloy"),   # Text To Speech
    )

    # Start the session
    await session.start(room=ctx.room, agent=assistant)

    # Send initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly and let them know you can see and hear them."
    )

    # Handle video processing in the background
    async def process_video():
        """Process video frames and store the latest image."""
        try:
            while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                try:
                    video_track = await get_video_track(ctx.room)
                    if video_track is None:
                        print("No video track found, waiting...")
                        await asyncio.sleep(2)
                        continue
                    
                    print("Starting video stream processing...")
                    async for event in rtc.VideoStream(video_track):
                        # Update the latest image in the function context
                        function_ctx.update_image(event.frame)
                        
                except Exception as e:
                    print(f"Video processing error: {e}")
                    await asyncio.sleep(1)
        except Exception as e:
            print(f"Video processing task error: {e}")

    # Start video processing task
    video_task = asyncio.create_task(process_video())

    try:
        # Keep the agent running
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
    finally:
        video_task.cancel()
        try:
            await video_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))