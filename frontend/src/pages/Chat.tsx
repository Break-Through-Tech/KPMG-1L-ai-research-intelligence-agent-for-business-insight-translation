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
import { useEffect, useState } from "react";

export default function Chat() {
	const [input, setInput] = useState("");
	const [userMessages, setUserMessages] = useState([]);
	const [responses, setResponses] = useState([]);
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
					}),
				});

				if (!response.ok || !response.body) {
					throw new Error(`HTTP error, Status: ${response.status}`);
				}

				// Add user message
				setUserMessages((prev) => [...prev, query]);

				// Parse stream response
				const reader = response.body.getReader();
				const decoder = new TextDecoder();
				let done = false;

				while (!done) {
					const { done: doneReading, value } = await reader.read();
					done = doneReading;

					const chunk = decoder.decode(value, { stream: !done });

					setResponses((prev) => [...prev, chunk]);
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
				{userMessages.map((msg, index) => (
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

				{responses.map((msg, index) => (
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
