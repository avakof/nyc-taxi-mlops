import React, { useState, useEffect, useRef } from "react";
// Deep import, not the package barrel: streamlit-component-lib's index.js also
// re-exports withStreamlitConnection/StreamlitComponentBase from StreamlitReact.js,
// which is compiled against React 16 ("react": "^16.14.0" in its own package.json) and
// crashes React 19 ("A React Element from an older version of React was rendered") --
// streamlit.js itself has no React import at all, so importing only this file sidesteps
// that incompatibility instead of needing to downgrade React.
import { Streamlit } from "streamlit-component-lib/dist/streamlit";

// State meaning definitions and color ramp (distinct from brand orange #F58426)
const STATE_CONFIG = {
  0: { label: "OK", color: "#10B981", bg: "#ECFDF5", border: "#A7F3D0", icon: "✓" },
  1: { label: "WATCH", color: "#CA8A04", bg: "#FEFCE8", border: "#FEF08A", icon: "◉" },
  2: { label: "WARNING", color: "#C2410C", bg: "#FFF7ED", border: "#FFEDD5", icon: "⚠" },
  3: { label: "ELEVATED", color: "#DC2626", bg: "#FEF2F2", border: "#FECACA", icon: "⚡" },
  4: { label: "RETRAIN", color: "#991B1B", bg: "#450A0A", border: "#F87171", icon: "🛑" }
};

export default function DriftDashboardControls() {
  const [data, setData] = useState(null);
  const [weekIndex, setWeekIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [whatIfThreshold, setWhatIfThreshold] = useState(20);
  const [selectedZoneId, setSelectedZoneId] = useState(null);
  const [viewMode, setViewMode] = useState("table"); // "table" | "chart"
  const playTimerRef = useRef(null);
  const initializedRef = useRef(false);

  // Handshake with Streamlit's real protocol (via the library's plain Streamlit
  // singleton, not its React-coupled wrapper -- see the import comment above).
  useEffect(() => {
    const handleRender = (event) => {
      const newArgs = event.detail.args;
      setData(newArgs);
      // initialWeekIndex/initialZoneId are only meant to seed state once; later
      // re-renders (e.g. a Streamlit rerun) must not stomp on user interaction.
      if (!initializedRef.current) {
        initializedRef.current = true;
        if (newArgs.initialWeekIndex !== undefined) setWeekIndex(newArgs.initialWeekIndex);
        if (newArgs.initialZoneId !== undefined) setSelectedZoneId(newArgs.initialZoneId);
      }
    };
    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, handleRender);
    Streamlit.setComponentReady();
    return () => Streamlit.events.removeEventListener(Streamlit.RENDER_EVENT, handleRender);
  }, []);

  // Auto-resize iframe
  useEffect(() => {
    const notifyHeight = () => {
      const height = document.getElementById("root")?.scrollHeight || document.body.scrollHeight || 650;
      Streamlit.setFrameHeight(height + 30);
    };
    notifyHeight();
    const timeout = setTimeout(notifyHeight, 150);
    return () => clearTimeout(timeout);
  }, [data, weekIndex, viewMode, whatIfThreshold]);

  // Handle Play simulation auto-advancing
  useEffect(() => {
    if (!isPlaying) {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
      return;
    }

    const totalWeeks = data?.weeks?.length || 1;
    playTimerRef.current = setInterval(() => {
      setWeekIndex((prev) => {
        const next = prev + 1 >= totalWeeks ? 0 : prev + 1;
        notifyStreamlit(next, whatIfThreshold, selectedZoneId);
        return next;
      });
    }, 1600);

    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying, data, whatIfThreshold, selectedZoneId]);

  const notifyStreamlit = (newWeek, newThreshold, newZoneId) => {
    Streamlit.setComponentValue({
      weekIndex: newWeek,
      whatIfThreshold: newThreshold,
      selectedZoneId: newZoneId
    });
  };

  const handleWeekChange = (newIndex) => {
    setWeekIndex(newIndex);
    notifyStreamlit(newIndex, whatIfThreshold, selectedZoneId);
  };

  const handleThresholdChange = (val) => {
    setWhatIfThreshold(val);
    // Purely client-side UI filter: do not trigger expensive Streamlit Python reruns
  };

  const handleZoneSelect = (zId) => {
    const nextZone = selectedZoneId === zId ? null : zId;
    setSelectedZoneId(nextZone);
    // Purely client-side table highlight: do not trigger expensive Streamlit Python reruns
  };

  if (!data || !data.weeks || data.weeks.length === 0) {
    return (
      <div style={{ padding: "20px", color: "#666", fontFamily: "sans-serif" }}>
        Loading drift monitoring controls...
      </div>
    );
  }

  const currentWeek = data.weeks[weekIndex] || data.weeks[0];
  const kpi = currentWeek.kpi || {};
  const zones = currentWeek.zones || [];
  const totalZones = zones.length || 20;
  const warningZones = zones.filter((z) => (z.state_pickups_residual || 0) >= 2);
  const warningPct = totalZones > 0 ? Math.round((warningZones.length / totalZones) * 100) : 0;
  const whatIfTriggered = warningPct >= whatIfThreshold;

  const maxStateCfg = STATE_CONFIG[kpi.max_state ?? 0] || STATE_CONFIG[0];

  return (
    <div
      style={{
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        color: "#1F2937",
        background: "#FFFFFF",
        padding: "16px 20px",
        borderRadius: "12px",
        border: "1px solid #E5E7EB",
        boxShadow: "0 1px 3px rgba(0,0,0,0.05)"
      }}
    >
      {/* 1. TOP HEADER & WEEKLY SIMULATION CONTROL */}
      <div
        style={{
          background: "linear-gradient(135deg, #006BB6 0%, #004D84 100%)",
          borderRadius: "10px",
          padding: "16px 20px",
          color: "#FFFFFF",
          marginBottom: "20px",
          boxShadow: "0 4px 6px -1px rgba(0, 107, 182, 0.2)"
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "12px",
            marginBottom: "14px"
          }}
        >
          <div>
            <div style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", letterSpacing: "1px", color: "#F58426" }}>
              Simulation Cadence & Trailing 168h Windows
            </div>
            <div style={{ fontSize: "18px", fontWeight: "800", marginTop: "2px" }}>
              Evaluation Week: {currentWeek.label || `Week ${weekIndex + 1}`}
            </div>
            <div style={{ fontSize: "12px", color: "#E0E7FF", marginTop: "2px" }}>
              Window End: {currentWeek.window_end_str} (America/New_York)
            </div>
          </div>

          {/* Controls */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <button
              onClick={() => handleWeekChange(Math.max(0, weekIndex - 1))}
              disabled={weekIndex === 0}
              style={{
                background: "rgba(255,255,255,0.15)",
                border: "1px solid rgba(255,255,255,0.3)",
                color: "#FFFFFF",
                borderRadius: "6px",
                padding: "6px 12px",
                cursor: weekIndex === 0 ? "not-allowed" : "pointer",
                fontWeight: "600",
                fontSize: "13px"
              }}
            >
              ⏮ Prev
            </button>

            <button
              onClick={() => setIsPlaying(!isPlaying)}
              style={{
                background: isPlaying ? "#EF4444" : "#F58426",
                border: "none",
                color: "#FFFFFF",
                borderRadius: "6px",
                padding: "8px 18px",
                cursor: "pointer",
                fontWeight: "700",
                fontSize: "13px",
                display: "flex",
                alignItems: "center",
                gap: "6px",
                boxShadow: "0 2px 4px rgba(0,0,0,0.2)"
              }}
            >
              {isPlaying ? "⏸ Pause" : "▶ Play Simulation"}
            </button>

            <button
              onClick={() => handleWeekChange(Math.min(data.weeks.length - 1, weekIndex + 1))}
              disabled={weekIndex === data.weeks.length - 1}
              style={{
                background: "rgba(255,255,255,0.15)",
                border: "1px solid rgba(255,255,255,0.3)",
                color: "#FFFFFF",
                borderRadius: "6px",
                padding: "6px 12px",
                cursor: weekIndex === data.weeks.length - 1 ? "not-allowed" : "pointer",
                fontWeight: "600",
                fontSize: "13px"
              }}
            >
              Next ⏭
            </button>
          </div>
        </div>

        {/* Slider */}
        <div style={{ marginTop: "8px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#BFDBFE", marginBottom: "4px" }}>
            <span>Start: {data.weeks[0]?.label || "Week 1"}</span>
            <span>Week {weekIndex + 1} of {data.weeks.length}</span>
            <span>End: {data.weeks[data.weeks.length - 1]?.label || `Week ${data.weeks.length}`}</span>
          </div>
          <input
            type="range"
            min={0}
            max={data.weeks.length - 1}
            value={weekIndex}
            onChange={(e) => handleWeekChange(parseInt(e.target.value, 10))}
            style={{
              width: "100%",
              cursor: "pointer",
              accentColor: "#F58426",
              height: "6px"
            }}
          />
        </div>
      </div>

      {/* 2. KPI CARDS (TOP OF PAGE) */}
      <div style={{ marginBottom: "24px" }}>
        <div style={{ fontSize: "13px", fontWeight: "700", color: "#006BB6", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>
          Citywide Drift Status (Trailing 168h Window)
        </div>

        {/* Main 3 High-Level KPI Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "12px", marginBottom: "14px" }}>
          {/* Max State Card */}
          <div
            style={{
              background: "#F9FAFB",
              border: `2px solid ${maxStateCfg.color}`,
              borderRadius: "8px",
              padding: "14px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between"
            }}
          >
            <div style={{ fontSize: "12px", color: "#6B7280", fontWeight: "600" }}>Overall Citywide State</div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", margin: "8px 0" }}>
              <span
                style={{
                  background: maxStateCfg.color,
                  color: "#FFFFFF",
                  fontSize: "13px",
                  fontWeight: "800",
                  padding: "4px 10px",
                  borderRadius: "20px",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "4px"
                }}
              >
                <span>{maxStateCfg.icon}</span>
                <span>STATE {kpi.max_state ?? 0}: {maxStateCfg.label}</span>
              </span>
            </div>
            <div style={{ fontSize: "11px", color: "#4B5563" }}>
              Worst dimension state across all 4 monitored signals
            </div>
          </div>

          {/* Weighted Score Card */}
          <div
            style={{
              background: "#F9FAFB",
              border: "1px solid #E5E7EB",
              borderRadius: "8px",
              padding: "14px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between"
            }}
          >
            <div style={{ fontSize: "12px", color: "#6B7280", fontWeight: "600" }}>Eta² Weighted Score</div>
            <div style={{ fontSize: "24px", fontWeight: "800", color: "#006BB6", margin: "4px 0" }}>
              {(kpi.weighted_score ?? 0).toFixed(3)}
              <span style={{ fontSize: "12px", color: "#9CA3AF", fontWeight: "500", marginLeft: "4px" }}>/ 4.0</span>
            </div>
            {/* Visual Gauge Bar */}
            <div style={{ background: "#E5E7EB", borderRadius: "4px", height: "8px", width: "100%", overflow: "hidden" }}>
              <div
                style={{
                  background: maxStateCfg.color,
                  height: "100%",
                  width: `${Math.min(100, ((kpi.weighted_score ?? 0) / 4) * 100)}%`,
                  transition: "width 0.3s ease"
                }}
              />
            </div>
            <div style={{ fontSize: "11px", color: "#6B7280", marginTop: "4px" }}>
              Reference variance-weighted average
            </div>
          </div>

          {/* Persistent Alert Card */}
          <div
            style={{
              background: kpi.persistent_alert ? "#FEF2F2" : "#F0FDF4",
              border: `1px solid ${kpi.persistent_alert ? "#FECACA" : "#BBF7D0"}`,
              borderRadius: "8px",
              padding: "14px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between"
            }}
          >
            <div style={{ fontSize: "12px", color: "#6B7280", fontWeight: "600" }}>Persistence Escalation</div>
            <div style={{ margin: "6px 0" }}>
              {kpi.persistent_alert ? (
                <span
                  style={{
                    background: "#DC2626",
                    color: "#FFFFFF",
                    fontSize: "13px",
                    fontWeight: "800",
                    padding: "4px 12px",
                    borderRadius: "6px",
                    display: "inline-block"
                  }}
                >
                  🚨 PERSISTENT ALERT
                </span>
              ) : (
                <span
                  style={{
                    background: "#16A34A",
                    color: "#FFFFFF",
                    fontSize: "13px",
                    fontWeight: "700",
                    padding: "4px 12px",
                    borderRadius: "6px",
                    display: "inline-block"
                  }}
                >
                  ✓ SYSTEM CALM
                </span>
              )}
            </div>
            <div style={{ fontSize: "11px", color: "#4B5563" }}>
              {kpi.persistent_alert
                ? "State >= 2 held for >= 3 consecutive windows"
                : "No sustained warning across consecutive windows"}
            </div>
          </div>
        </div>

        {/* 4 Small Sub-Cards (One Per Dimension) */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "10px" }}>
          {/* 1. Pickups Residual */}
          <div
            style={{
              background: "#FFFFFF",
              border: `1px solid ${STATE_CONFIG[kpi.state_pickups_residual ?? 0]?.border || "#E5E7EB"}`,
              borderLeft: `4px solid ${STATE_CONFIG[kpi.state_pickups_residual ?? 0]?.color || "#10B981"}`,
              borderRadius: "6px",
              padding: "10px 12px"
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "12px", fontWeight: "700", color: "#111827" }}>Pickups Residual</span>
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: "700",
                  padding: "2px 6px",
                  borderRadius: "4px",
                  background: STATE_CONFIG[kpi.state_pickups_residual ?? 0]?.color,
                  color: "#FFF"
                }}
              >
                S{kpi.state_pickups_residual ?? 0}: {STATE_CONFIG[kpi.state_pickups_residual ?? 0]?.label}
              </span>
            </div>
            <div style={{ fontSize: "11px", color: "#4B5563", marginTop: "6px" }}>
              PSI: <b>{(kpi.psi_pickups_residual ?? 0).toFixed(4)}</b> | d: <b>{(kpi.cohens_d_pickups_residual ?? 0).toFixed(4)}</b>
            </div>
            <div style={{ fontSize: "10px", color: "#6B7280", marginTop: "2px" }}>
              ADF: <span style={{ fontWeight: "600", color: "#006BB6" }}>{kpi.adf_label || "reverting"}</span> (confirmatory)
            </div>
          </div>

          {/* 2. Zone Mix */}
          <div
            style={{
              background: "#FFFFFF",
              border: `1px solid ${STATE_CONFIG[kpi.state_zone_mix ?? 0]?.border || "#E5E7EB"}`,
              borderLeft: `4px solid ${STATE_CONFIG[kpi.state_zone_mix ?? 0]?.color || "#10B981"}`,
              borderRadius: "6px",
              padding: "10px 12px"
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "12px", fontWeight: "700", color: "#111827" }}>Zone Mix</span>
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: "700",
                  padding: "2px 6px",
                  borderRadius: "4px",
                  background: STATE_CONFIG[kpi.state_zone_mix ?? 0]?.color,
                  color: "#FFF"
                }}
              >
                S{kpi.state_zone_mix ?? 0}: {STATE_CONFIG[kpi.state_zone_mix ?? 0]?.label}
              </span>
            </div>
            <div style={{ fontSize: "11px", color: "#4B5563", marginTop: "6px" }}>
              PSI: <b>{(kpi.psi_zone_mix ?? 0).toFixed(4)}</b> | w: <b>{(kpi.cohens_w_zone_mix ?? 0).toFixed(4)}</b>
            </div>
            <div style={{ fontSize: "10px", color: "#6B7280", marginTop: "2px" }}>
              Share vector across 20 modeled zones
            </div>
          </div>

          {/* 3. Hour of Day Mix */}
          <div
            style={{
              background: "#FFFFFF",
              border: `1px solid ${STATE_CONFIG[kpi.state_hour_of_day_mix ?? 0]?.border || "#E5E7EB"}`,
              borderLeft: `4px solid ${STATE_CONFIG[kpi.state_hour_of_day_mix ?? 0]?.color || "#10B981"}`,
              borderRadius: "6px",
              padding: "10px 12px"
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "12px", fontWeight: "700", color: "#111827" }}>Hour-of-Day Mix</span>
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: "700",
                  padding: "2px 6px",
                  borderRadius: "4px",
                  background: STATE_CONFIG[kpi.state_hour_of_day_mix ?? 0]?.color,
                  color: "#FFF"
                }}
              >
                S{kpi.state_hour_of_day_mix ?? 0}: {STATE_CONFIG[kpi.state_hour_of_day_mix ?? 0]?.label}
              </span>
            </div>
            <div style={{ fontSize: "11px", color: "#4B5563", marginTop: "6px" }}>
              PSI: <b>{(kpi.psi_hour_of_day_mix ?? 0).toFixed(4)}</b> | w: <b>{(kpi.cohens_w_hour_of_day_mix ?? 0).toFixed(4)}</b>
            </div>
            <div style={{ fontSize: "10px", color: "#6B7280", marginTop: "2px" }}>
              24-hour diurnal pickup distribution
            </div>
          </div>

          {/* 4. Weekend Mix */}
          <div
            style={{
              background: "#FFFFFF",
              border: `1px solid ${STATE_CONFIG[kpi.state_weekend_mix ?? 0]?.border || "#E5E7EB"}`,
              borderLeft: `4px solid ${STATE_CONFIG[kpi.state_weekend_mix ?? 0]?.color || "#10B981"}`,
              borderRadius: "6px",
              padding: "10px 12px"
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "12px", fontWeight: "700", color: "#111827" }}>Weekend Mix</span>
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: "700",
                  padding: "2px 6px",
                  borderRadius: "4px",
                  background: STATE_CONFIG[kpi.state_weekend_mix ?? 0]?.color,
                  color: "#FFF"
                }}
              >
                S{kpi.state_weekend_mix ?? 0}: {STATE_CONFIG[kpi.state_weekend_mix ?? 0]?.label}
              </span>
            </div>
            <div style={{ fontSize: "11px", color: "#4B5563", marginTop: "6px" }}>
              PSI: <b>{(kpi.psi_weekend_mix ?? 0).toFixed(4)}</b> | h: <b>{(kpi.cohens_h_weekend_mix ?? 0).toFixed(4)}</b>
            </div>
            <div style={{ fontSize: "10px", color: "#6B7280", marginTop: "2px" }}>
              Weekday vs Weekend demand share
            </div>
          </div>
        </div>
      </div>

      {/* 3. ZONE VIEW (REFLECTS SELECTED WEEK) */}
      <div style={{ borderTop: "2px solid #F3F4F6", paddingTop: "18px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px", marginBottom: "12px" }}>
          <div>
            <div style={{ fontSize: "13px", fontWeight: "700", color: "#006BB6", textTransform: "uppercase", letterSpacing: "0.5px" }}>
              Zone-Level Residual Drift (Weekly Breakdown)
            </div>
            <div style={{ fontSize: "12px", color: "#6B7280", marginTop: "2px" }}>
              Evaluates pickups_residual distribution per zone over the selected week
            </div>
          </div>

          {/* Table / Chart Toggle */}
          <div style={{ display: "flex", background: "#F3F4F6", borderRadius: "6px", padding: "2px" }}>
            <button
              onClick={() => setViewMode("table")}
              style={{
                background: viewMode === "table" ? "#006BB6" : "transparent",
                color: viewMode === "table" ? "#FFF" : "#4B5563",
                border: "none",
                borderRadius: "4px",
                padding: "4px 10px",
                fontSize: "12px",
                fontWeight: "600",
                cursor: "pointer"
              }}
            >
              Table View
            </button>
            <button
              onClick={() => setViewMode("chart")}
              style={{
                background: viewMode === "chart" ? "#006BB6" : "transparent",
                color: viewMode === "chart" ? "#FFF" : "#4B5563",
                border: "none",
                borderRadius: "4px",
                padding: "4px 10px",
                fontSize: "12px",
                fontWeight: "600",
                cursor: "pointer"
              }}
            >
              Bar Chart View
            </button>
          </div>
        </div>

        {/* Headline KPI + What-If Control */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "12px",
            background: "#F8FAFC",
            border: "1px solid #E2E8F0",
            borderRadius: "8px",
            padding: "14px",
            marginBottom: "16px"
          }}
        >
          {/* Headline KPI */}
          <div>
            <div style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "#64748B" }}>
              Headline Warning Proportion
            </div>
            <div style={{ fontSize: "20px", fontWeight: "800", color: warningZones.length > 0 ? "#C2410C" : "#16A34A", marginTop: "4px" }}>
              {warningZones.length} of {totalZones} zones ({warningPct}%)
            </div>
            <div style={{ fontSize: "12px", color: "#475569", marginTop: "2px" }}>
              At warning (state &gt;= 2) or above this week
            </div>
          </div>

          {/* Interactive What-If Control */}
          <div style={{ borderLeft: "2px solid #E2E8F0", paddingLeft: "14px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "12px", fontWeight: "700", color: "#006BB6" }}>
                Exploratory What-If Threshold:
              </span>
              <span
                style={{
                  background: whatIfTriggered ? "#FEE2E2" : "#DCFCE7",
                  color: whatIfTriggered ? "#991B1B" : "#166534",
                  fontSize: "11px",
                  fontWeight: "700",
                  padding: "2px 8px",
                  borderRadius: "4px"
                }}
              >
                {whatIfTriggered ? `ALERT TRIGGERED (>= ${whatIfThreshold}%)` : `CALM (< ${whatIfThreshold}%)`}
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "6px" }}>
              <input
                type="range"
                min={5}
                max={50}
                step={5}
                value={whatIfThreshold}
                onChange={(e) => handleThresholdChange(parseInt(e.target.value, 10))}
                style={{ flex: 1, accentColor: "#F58426", cursor: "pointer" }}
              />
              <span style={{ fontSize: "13px", fontWeight: "700", color: "#1F2937", minWidth: "35px" }}>
                {whatIfThreshold}%
              </span>
            </div>

            <div style={{ fontSize: "10px", color: "#64748B", marginTop: "4px", fontStyle: "italic" }}>
              Exploratory/informational only — does not modify citywide decision or underlying reports.
            </div>
          </div>
        </div>

        {/* Zone Visualization (Table or Bar Chart) */}
        {viewMode === "table" ? (
          <div style={{ overflowX: "auto", border: "1px solid #E5E7EB", borderRadius: "8px" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", textAlign: "left" }}>
              <thead>
                <tr style={{ background: "#F9FAFB", borderBottom: "1px solid #E5E7EB", color: "#4B5563" }}>
                  <th style={{ padding: "8px 12px" }}>Zone</th>
                  <th style={{ padding: "8px 12px" }}>Borough</th>
                  <th style={{ padding: "8px 12px" }}>State</th>
                  <th style={{ padding: "8px 12px" }}>z-score</th>
                  <th style={{ padding: "8px 12px" }}>Cohen's d</th>
                  <th style={{ padding: "8px 12px" }}>PSI</th>
                  <th style={{ padding: "8px 12px" }}>Alert</th>
                </tr>
              </thead>
              <tbody>
                {zones.map((z) => {
                  const zState = z.state_pickups_residual ?? 0;
                  const cfg = STATE_CONFIG[zState] || STATE_CONFIG[0];
                  const isSelected = selectedZoneId === z.zone_id;

                  return (
                    <tr
                      key={z.zone_id}
                      onClick={() => handleZoneSelect(z.zone_id)}
                      style={{
                        borderBottom: "1px solid #F3F4F6",
                        background: isSelected ? "#EFF6FF" : "transparent",
                        cursor: "pointer",
                        transition: "background 0.15s ease"
                      }}
                    >
                      <td style={{ padding: "8px 12px", fontWeight: "600", color: isSelected ? "#006BB6" : "#111827" }}>
                        {z.name || `Zone ${z.zone_id}`}
                      </td>
                      <td style={{ padding: "8px 12px", color: "#6B7280" }}>{z.borough || "-"}</td>
                      <td style={{ padding: "8px 12px" }}>
                        <span
                          style={{
                            background: cfg.color,
                            color: "#FFFFFF",
                            padding: "2px 6px",
                            borderRadius: "4px",
                            fontWeight: "700",
                            fontSize: "11px"
                          }}
                        >
                          {zState}: {cfg.label}
                        </span>
                      </td>
                      <td style={{ padding: "8px 12px", fontFamily: "monospace" }}>
                        {(z.z_pickups_residual ?? 0) >= 0 ? `+${(z.z_pickups_residual ?? 0).toFixed(2)}` : (z.z_pickups_residual ?? 0).toFixed(2)}
                      </td>
                      <td style={{ padding: "8px 12px", fontFamily: "monospace" }}>
                        {(z.cohens_d_pickups_residual ?? 0).toFixed(3)}
                      </td>
                      <td style={{ padding: "8px 12px", fontFamily: "monospace" }}>
                        {(z.psi_pickups_residual ?? 0).toFixed(4)}
                      </td>
                      <td style={{ padding: "8px 12px" }}>
                        {z.persistent_alert ? (
                          <span style={{ color: "#DC2626", fontWeight: "700" }}>ALERT</span>
                        ) : (
                          <span style={{ color: "#9CA3AF" }}>Calm</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          /* Bar Chart View */
          <div style={{ border: "1px solid #E5E7EB", borderRadius: "8px", padding: "14px", background: "#FAFAFA" }}>
            <div style={{ fontSize: "11px", fontWeight: "600", color: "#6B7280", marginBottom: "8px" }}>
              Cohen's d Residual Impact by Zone (Color = Drift State)
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              {zones
                .slice()
                .sort((a, b) => Math.abs(b.cohens_d_pickups_residual || 0) - Math.abs(a.cohens_d_pickups_residual || 0))
                .map((z) => {
                  const val = z.cohens_d_pickups_residual || 0;
                  const cfg = STATE_CONFIG[z.state_pickups_residual || 0] || STATE_CONFIG[0];
                  const barWidth = Math.min(100, Math.abs(val) * 200);

                  return (
                    <div
                      key={z.zone_id}
                      onClick={() => handleZoneSelect(z.zone_id)}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        fontSize: "11px",
                        cursor: "pointer",
                        padding: "2px 4px",
                        borderRadius: "4px",
                        background: selectedZoneId === z.zone_id ? "#EFF6FF" : "transparent"
                      }}
                    >
                      <div style={{ width: "160px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontWeight: "600" }}>
                        {z.name || `Zone ${z.zone_id}`}
                      </div>
                      <div style={{ flex: 1, display: "flex", alignItems: "center", height: "14px", background: "#E5E7EB", borderRadius: "3px", overflow: "hidden" }}>
                        <div
                          style={{
                            height: "100%",
                            width: `${Math.max(4, barWidth)}%`,
                            background: cfg.color
                          }}
                        />
                      </div>
                      <div style={{ width: "45px", textAlign: "right", fontFamily: "monospace", color: "#4B5563" }}>
                        {val.toFixed(2)}
                      </div>
                      <div style={{ width: "65px" }}>
                        <span style={{ fontSize: "10px", fontWeight: "700", color: cfg.color }}>
                          State {z.state_pickups_residual || 0}
                        </span>
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
