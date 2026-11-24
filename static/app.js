(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const box = $("review"), btn = $("analyzeBtn"), err = $("error"), res = $("result");

  const fillList = (el, items, emptyText) => {
    el.innerHTML = "";
    (items.length ? items : [emptyText]).forEach((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      if (!items.length) li.className = "none";
      el.appendChild(li);
    });
  };
  const fillChips = (el, items) => {
    el.innerHTML = "";
    if (!items.length) { el.textContent = "-"; return; }
    items.forEach((w) => {
      const s = document.createElement("span");
      s.className = "tag"; s.textContent = w; el.appendChild(s);
    });
  };

  async function analyze() {
    err.classList.add("hidden");
    const text = box.value.trim();
    if (!text) { showError("Please paste a review first."); return; }
    btn.disabled = true; btn.textContent = "Analyzing...";
    try {
      const r = await fetch("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Something went wrong.");
      render(d);
    } catch (e) {
      showError(e.message || "Could not reach the server.");
    } finally {
      btn.disabled = false; btn.textContent = "Analyze review";
    }
  }

  function render(d) {
    const v = $("verdict");
    v.textContent = d.verdict;
    v.className = "badge " + (d.verdict === "Likely fake" ? "fake" : d.verdict === "Likely genuine" ? "real" : "unsure");
    $("pct").textContent = `${(d.fake_probability * 100).toFixed(1)}% chance of being fake`;
    $("fill").style.left = (d.fake_probability * 100).toFixed(1) + "%";
    fillList($("warn"), d.warning_signs, "None found");
    fillList($("good"), d.trust_signs, "None found");
    fillChips($("wFake"), d.words_pointing_to_fake);
    fillChips($("wReal"), d.words_pointing_to_genuine);
    res.classList.remove("hidden");
    res.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function showError(m) { err.textContent = m; err.classList.remove("hidden"); res.classList.add("hidden"); }

  box.addEventListener("input", () => { $("count").textContent = `${box.value.length} / 5000`; });
  btn.addEventListener("click", analyze);
  $("clearBtn").addEventListener("click", () => {
    box.value = ""; $("count").textContent = "0 / 5000"; res.classList.add("hidden"); err.classList.add("hidden"); box.focus();
  });
  document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => {
    box.value = c.dataset.ex; box.dispatchEvent(new Event("input")); analyze();
  }));
  box.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze(); });
})();
