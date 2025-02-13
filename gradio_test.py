import gradio as gr
import asyncio
import websockets
import json
import uuid

# Function to handle WebSocket communication
async def websocket_communication(user_message):
    uri = "ws://129.213.84.161:8000/ws/{client_id}"  # Replace with your FastAPI WebSocket endpoint
    client_id = str(uuid.uuid4())  # Generate a unique client_id
    async with websockets.connect(uri.format(client_id=client_id)) as websocket:
        # Send the user's message to the server
        await websocket.send(json.dumps({
            "event": "conversation.item.create",
            "data": {"content": user_message}
        }))
        # Wait for the server's response
        response = await websocket.recv()
        response_data = json.loads(response)
        return response_data.get("response", "No response received.")

# Wrapper function to run the async function
def gradio_interface(user_message):
    return asyncio.run(websocket_communication(user_message))

# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your message here..."),
    outputs=gr.Textbox(),
    title="Real-Time Chat with WebSocket",
    description="This interface communicates with a FastAPI WebSocket endpoint in real-time."
)

if __name__ == "__main__":
    iface.launch(share=True)
