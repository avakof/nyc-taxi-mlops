import React from "react";
import ReactDOM from "react-dom/client";
import DriftDashboardControls from "./Component";

// NOT using streamlit-component-lib's withStreamlitConnection/StreamlitComponentBase --
// that package pins "react": "^16.14.0" and its wrapper calls React.createElement with
// its own bundled React 16 copy, which React 19's reconciler rejects ("A React Element
// from an older version of React was rendered"). Component.jsx instead talks to the
// library's plain `Streamlit` singleton directly (streamlit.js has zero React import,
// so it's version-agnostic) and does its own componentReady/render-event handshake.
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <DriftDashboardControls />
  </React.StrictMode>
);
