import {
	AppBar,
	Box,
	Button,
	Paper,
	Stack,
	TextField,
	Toolbar,
	Typography,
} from "@mui/material";
import { type FormEvent, useEffect, useState } from "react";

const CHAT_HISTORY_KEY = "kpmg-ai-research-chat-history";

type ChatHistory = {
	userMessages: string[];
	responses: string[];
};

function loadChatHistory(): ChatHistory {
	try {
		const savedHistory = localStorage.getItem(CHAT_HISTORY_KEY);
		if (!savedHistory) {
			return { userMessages: [], responses: [] };
		}

		const parsed: unknown = JSON.parse(savedHistory);
		if (
			typeof parsed === "object" &&
			parsed !== null &&
			"userMessages" in parsed &&
			"responses" in parsed &&
			Array.isArray(parsed.userMessages) &&
			parsed.userMessages.every((message) => typeof message === "string") &&
			Array.isArray(parsed.responses) &&
			parsed.responses.every((message) => typeof message === "string")
		) {
			return {
				userMessages: parsed.userMessages,
				responses: parsed.responses,
			};
		}

		console.error("Saved chat history has an invalid format.");
	} catch (err) {
		console.error("Could not load saved chat history.", err);
	}

	return { userMessages: [], responses: [] };
}

export default function Chat() {
	const [input, setInput] = useState("");
	const [chatHistory, setChatHistory] = useState(loadChatHistory);

	useEffect(() => {
		try {
			localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatHistory));
		} catch (err) {
			console.error("Could not save chat history.", err);
		}
	}, [chatHistory]);

	// Make request to Chat Backend
	const handleSend = (e: FormEvent<HTMLFormElement>) => {
		e.preventDefault();

		const sendMessage = async () => {
			const query = input.trim();
			if (!query) return;

			try {
				setInput("");

				const response = await fetch("http://localhost:8000/api/chat", {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
					body: JSON.stringify({
						query: input,
					}),
				});

				if (!response.ok || !response.body) {
					throw new Error(`HTTP error, Status: ${response.status}`);
				}

				// Add user message
				setChatHistory((prev) => ({
					...prev,
					userMessages: [...prev.userMessages, query],
				}));

				// Parse stream response
				const reader = response.body.getReader();
				const decoder = new TextDecoder();
				let done = false;

				while (!done) {
					const { done: doneReading, value } = await reader.read();
					done = doneReading;

					const chunk = decoder.decode(value, { stream: !done });

					setChatHistory((prev) => ({
						...prev,
						responses: [...prev.responses, chunk],
					}));
				}
			} catch (err) {
				console.error(err);
			}
		};

		sendMessage();
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
				{chatHistory.userMessages.map((msg, index) => (
					<Paper
						key={index}
						elevation={0}
						sx={{
							ml: "auto",
							width: 100,
							backgroundColor: "secondary.main",
							p: 3,
							mb: 2,
						}}
					>
						<Typography align="left" sx={{ color: "white" }}>
							{msg}
						</Typography>
					</Paper>
				))}

				{chatHistory.responses.map((msg, index) => (
					<Typography key={index}>{msg}</Typography>
				))}
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
