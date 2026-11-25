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
import { useState } from "react";
import type React from "react";

type Message = {
	role: "user" | "assistant";
	content: string;
};

const INTERRUPTED_MESSAGE = "AI is busy, please retry shortly";

export default function Chat() {
	const [input, setInput] = useState("");
	const [messages, setMessages] = useState<Message[]>([]);
	const [status, setStatus] = useState("");
	const [isStreaming, setIsStreaming] = useState(false);
	const [failedQuery, setFailedQuery] = useState<{
		query: string;
		history: Message[];
	} | null>(null);

	// Make request to Chat Backend. `history` is the conversation before this
	// query; the user message and an empty assistant message are appended to it.
	const sendQuery = async (query: string, history: Message[]) => {
		setIsStreaming(true);
		setFailedQuery(null);
		setStatus("Connecting...");
		setMessages([
			...history,
			{ role: "user", content: query },
			{ role: "assistant", content: "" },
		]);

		let interrupted = false;

		const handleEvent = (event: any) => {
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
			} else if (event.type === "status") {
				setStatus(event.message);
			} else if (event.type === "error") {
				interrupted = true;
			}
		};

		try {
			const response = await fetch("http://localhost:8000/api/chat", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({ query, history }),
			});

			if (!response.ok || !response.body) {
				throw new Error(`HTTP error, Status: ${response.status}`);
			}

			// Parse stream response (events are separated by blank lines)
			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = "";
			let done = false;

			while (!done && !interrupted) {
				const { done: doneReading, value } = await reader.read();
				done = doneReading;
				buffer += decoder.decode(value, { stream: !done });

				const parts = buffer.split("\n\n");
				buffer = parts.pop() ?? "";
				for (const part of parts) {
					if (part.trim()) handleEvent(JSON.parse(part));
				}
			}
			if (buffer.trim() && !interrupted) handleEvent(JSON.parse(buffer));
		} catch (err) {
			console.error(err);
			interrupted = true;
		} finally {
			setStatus("");
			setIsStreaming(false);
			if (interrupted) setFailedQuery({ query, history });
		}
	};

	const handleSend = (e: React.FormEvent) => {
		e.preventDefault();
		const query = input.trim();
		if (!query || isStreaming) return;
		setInput("");
		sendQuery(query, messages);
	};

	const handleRetry = () => {
		if (!failedQuery || isStreaming) return;
		sendQuery(failedQuery.query, failedQuery.history);
	};

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

				{status && <Typography>{status}</Typography>}

				{failedQuery && (
					<Stack direction="row" spacing={2} alignItems="center">
						<Typography color="error">{INTERRUPTED_MESSAGE}</Typography>
						<Button variant="outlined" onClick={handleRetry}>
							Retry
						</Button>
					</Stack>
				)}
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
                    bgcolor: "#f5f5ff"
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

				<Button type="submit" variant="contained" disabled={isStreaming}>
					Send
				</Button>
			</Box>
		</Stack>
	);
}
