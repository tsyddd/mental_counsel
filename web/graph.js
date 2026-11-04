(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const svgEl = (tag, attrs = {}) => {
    const element = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, String(value));
    return element;
  };
  const short = (value, length = 14) => value.length > length ? value.slice(0, length) + '…' : value;

  window.renderMemoryGraph = function (data, target, detail) {
    target.replaceChildren();
    const events = (data.memories || []).filter(item => item.layer === 'episodic')
      .sort((a, b) => new Date(a.created_at) - new Date(b.created_at)).slice(-18);
    const eventIds = new Set(events.map(item => item.id));
    const concepts = (data.concepts || []).filter(item =>
      (item.memory_ids || []).some(id => eventIds.has(id))).slice(0, 9);
    const conceptIds = new Set(concepts.map(item => item.id));
    const intents = (data.intents || []).filter(item => !['superseded','rejected'].includes(item.status)).slice(0, 8);
    if (!events.length && !concepts.length && !intents.length) {
      const empty = document.createElement('p');
      empty.className = 'muted';
      empty.textContent = '还没有情景事件。继续交流后，事件和意图会显示在这里。';
      target.append(empty);
      return;
    }
    const height = Math.max(390, events.length * 66 + 100, concepts.length * 112 + 95,
      intents.length * 132 + 95);
    const positions = new Map();
    events.forEach((item, index) => positions.set(item.id, { x: 135, y: 90 + index * 66 }));
    concepts.forEach((item, index) => positions.set(item.id, { x: 465, y: 100 + index * 112 }));
    intents.forEach((item, index) => positions.set(item.id, { x: 795, y: 112 + index * 132 }));
    const svg = svgEl('svg', { viewBox: `0 0 950 ${height}`, width: 950, height,
      role: 'img', 'aria-label': '情景事件和意图的关系图' });
    const header = svgEl('g', { class: 'graph-headers' });
    [['情景事件 · 时间顺序', 38], ['待确认意图', 700]].forEach(([label, x]) => {
      const title = svgEl('text', { x, y: 35 }); title.textContent = label; header.append(title);
    });
    svg.append(header);
    const edges = svgEl('g', { class: 'graph-edges' });
    const links = [];
    const drawEdge = (source, targetId, kind) => {
      const from = positions.get(source), to = positions.get(targetId);
      if (!from || !to) return;
      const curve = kind === 'time'
        ? `M ${from.x} ${from.y + 19} C ${from.x - 34} ${from.y + 35}, ${to.x - 34} ${to.y - 35}, ${to.x} ${to.y - 19}`
        : `M ${from.x + 104} ${from.y} C ${from.x + 215} ${from.y}, ${to.x - 215} ${to.y}, ${to.x - 104} ${to.y}`;
      const path = svgEl('path', { d: curve, class: `graph-edge ${kind}` });
      edges.append(path); links.push({ source, target: targetId, path });
    };
    for (let i = 1; i < events.length; i++) drawEdge(events[i - 1].id, events[i].id, 'time');
    for (const concept of concepts) for (const eventId of concept.memory_ids || [])
      drawEdge(eventId, concept.id, 'relation');
    for (const intent of intents) for (const eventId of intent.support_event_ids || [])
      drawEdge(eventId, intent.id, 'relation');
    svg.append(edges);
    const nodes = svgEl('g', { class: 'graph-nodes' });
    const addNode = (item, kind, label, sub) => {
      const { x, y } = positions.get(item.id);
      const group = svgEl('g', { class: `graph-node ${kind}`, transform: `translate(${x},${y})`,
        tabindex: 0, role: 'button', 'aria-label': `${label}：${item.content || item.label}` });
      group.dataset.id = item.id;
      group.append(svgEl('rect', { x: -106, y: -25, width: 212, height: 50, rx: 12 }));
      const text = svgEl('text', { x: -94, y: -4 }); text.textContent = short(item.content || item.label);
      const caption = svgEl('text', { x: -94, y: 14, class: 'graph-node-sub' }); caption.textContent = sub;
      group.append(text, caption);
      const select = () => {
        nodes.querySelectorAll('.graph-node').forEach(node => node.classList.toggle('selected', node.dataset.id === item.id));
        links.forEach(link => link.path.classList.toggle('active', link.source === item.id || link.target === item.id));
        detail.textContent = `${label}：${item.content || item.label}${sub ? '\n' + sub : ''}`;
      };
      group.addEventListener('click', select);
      group.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } });
      nodes.append(group);
    };
    events.forEach(item => addNode(item, 'event', '事件', `会话 ${String(item.source_session).slice(0,8)} · 第${item.start_turn}-${item.end_turn}轮` ));
    concepts.forEach(item => addNode(item, 'concept', '主题', `${item.support_count} 条证据`));
    intents.forEach(item => addNode(item, 'intent', '意图', `${item.status} · 第${item.extracted_turn}轮形成` ));
    svg.append(nodes);
    target.append(svg);
    detail.textContent = '点击节点查看完整内容与相连证据。虚线表示事件发生顺序，实线表示事件对意图的支持关系。';
  };
})();
