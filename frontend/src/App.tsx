import { useState } from "react";
import "./App.css";
import Chat from "./pages/Chat";
import { Box, createTheme, ThemeProvider } from "@mui/material";

const theme = createTheme({
	palette: {
		primary: {
			// Using an imported MUI color object automatically calculates .light and .dark shades
			main: "#00338D",
		},
    secondary: {
      main: "#005EB8"
    }
	},
});

function App() {
	return (
		<ThemeProvider theme={theme}>
			<Box
				sx={{
					display: "flex",
					justifyContent: "center", // Centers horizontally
					alignItems: "center",
					minHeight: "100vh",
					width: "100%",
				}}
			>
				<Chat />
			</Box>
		</ThemeProvider>
	);
}

export default App;
