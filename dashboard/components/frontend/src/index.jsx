import React from "react";
import ReactDOM from "react-dom/client";
import { withStreamlitConnection } from "streamlit-component-lib";
import DriftDashboardControls from "./Component";

// withStreamlitConnection owns the handshake (componentReady, the render-event
// listener, args/theme as props) -- Component.jsx itself only calls
// Streamlit.setFrameHeight()/setComponentValue() for its own updates.
const ConnectedDriftDashboardControls = withStreamlitConnection(DriftDashboardControls);

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <ConnectedDriftDashboardControls />
  </React.StrictMode>
);
