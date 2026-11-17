import {
	AppBar,
	Avatar,
	Box,
	Button,
	Paper,
	Stack,
	TextField,
	Toolbar,
	Typography,
} from "@mui/material";
import MuiMarkdown from "mui-markdown";
import { useEffect, useState } from "react";

type Message = {
	role: "user" | "assistant";
	content: string;
};

export default function Chat() {
	const [input, setInput] = useState("");
	const [messages, setMessages] = useState<Message[]>([]);
	const [status, setStatus] = useState("");

	// Make request to Chat Backend
	const handleSend = (e) => {
		e.preventDefault();

		const sendMessage = async () => {
			const query = input.trim();
			if (!query) return;

			setStatus("Connecting...");

			try {
				setInput("");

				const response = await fetch("http://localhost:8000/api/chat", {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
					body: JSON.stringify({
						query: input,
                        history: messages
					}),
				});

				if (!response.ok || !response.body) {
					throw new Error(`HTTP error, Status: ${response.status}`);
				}

				// Add user message
				setMessages((prev) => [
						...prev,
						{ role: "user", content: query },
						{ role: "assistant", content: "" },
					]);

				// Parse stream response
				const reader = response.body.getReader();
				const decoder = new TextDecoder();
				let done = false;

				while (!done) {
					const { done: doneReading, value } = await reader.read();
					done = doneReading;

					const chunk = decoder.decode(value, { stream: !done });

                    const event = JSON.parse(chunk)

                    if (event.type === "chunk") {
                        setMessages((prev) => {
                            const updated = [...prev];
                            const last = updated[updated.length - 1];
                            updated[updated.length - 1] = {
                                ...last,
                                content: last.content + event.content,
                            };
                            return updated;
                        });
                    } else if (event.type == "status") {
                        setStatus(event.message)
                        console.log(event.message)
                    }
				}
			} catch (err) {
				console.error(err);
			} finally {
                setStatus("")
            }
		};

		sendMessage();
	};

    useEffect(() => {
        console.log(messages)
    }, [messages])

	return (
		<Stack sx={{ minHeight: "100dvh" }}>
			<AppBar position="fixed">
				<Toolbar>
					<Typography variant="h6">KPMG AI Research Intelligence</Typography>
				</Toolbar>
			</AppBar>

			{/* Space for the fixed app bar */}
			<Toolbar />

			<Box
				sx={{
					width: "80%",
					mx: "auto",
					pt: 2,
					pb: "120px", // Space for the fixed input
				}}
			>
				{messages.map((msg, index) =>
					msg.role === "user" ? (
						<Paper
							key={index}
							elevation={0}
							sx={{
								ml: "auto",
								width: "fit-content",
								backgroundColor: "secondary.main",
								p: 3,
								mb: 2,
							}}
						>
							<Typography align="left" sx={{ color: "white" }}>
								{msg.content}
							</Typography>
						</Paper>
					) : (
						// <Typography key={index} align="left" sx={{ mb: 2 }}>
						// 	{msg.content}
						// </Typography>
                        <Box sx = {{
                            textAlign: 'left',
                            width: '100%'
                        }}>
                            <MuiMarkdown copiable>
                                {msg.content}
                            </MuiMarkdown>
                        </Box>
					),
				)}

				{status &&
                    <Typography>{status}</Typography>
                }
			</Box>

			<Box
				component="form"
				onSubmit={handleSend}
				sx={{
					position: "fixed",
					bottom: 0,
					left: "10%",
					width: "80%",
					display: "flex",
					gap: 2,
					p: 2,
					boxSizing: "border-box",
					zIndex: 1100,
                    bgcolor: "#e8e8ff"
				}}
			>
				<TextField
					fullWidth
					label="How can I help today?"
					value={input}
					onChange={(event) => setInput(event.target.value)}
                    sx={{
                        backgroundColor: "white"
                    }}
				/>

				<Button type="submit" variant="contained">
					Send
				</Button>
			</Box>
		</Stack>
	);
}
