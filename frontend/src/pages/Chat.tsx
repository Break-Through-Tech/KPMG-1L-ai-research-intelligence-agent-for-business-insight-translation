import { Button, TextField, Typography } from "@mui/material";
import { useEffect, useState } from "react";

export default function Chat() {
	const [input, setInput] = useState("");
    const [responses, setResponses] = useState([]);
    const [status, setStatus] = useState("");

    // Make request to Chat Backend
	const handleSend = () => {
		const sendMessage = async () => {

            const query = input.trim();
            if (!query) return;

            setStatus("Connecting...")

			try {
				const response = await fetch("http://localhost:8000/api/chat", {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: input
                    })
                });

                if (!response.ok || !response.body) {
                    throw new Error(`HTTP error, Status: ${response.status}`)
                }

                // Parse stream response
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let done = false;


                while (!done) {
                    const {done: doneReading, value} = await reader.read();
                    done = doneReading;

                    const chunk = decoder.decode(value, { stream: !done });

                    setResponses(prev => [...prev, chunk])

                }

                const data = await response.json()
                console.log(data)


				setInput("");
			} catch (err) {
				console.error(err);
			}
		};

        sendMessage()
	};

	return (
		<>
			<Typography variant="h4">KPMG AI Research Intelligence</Typography>

			<TextField
				label="How can I help today?"
				value={input}
				onChange={(event) => setInput(event.target.value)}
			/>

			<Button onClick={handleSend}>Send</Button>
            {
                responses && (
                    responses.map(msg => (
                        <Typography>
                            {msg}
                        </Typography>
                    ))
                )
            }
		</>
	);
}
