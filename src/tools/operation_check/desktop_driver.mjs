// Actual desktop input driver. Inject the supported @oai/sky client from the
// computer-use runtime. Each act() performs ONE OS action and then refreshes;
// the caller inspects that observation before deciding the next action.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';

export class DesktopDriver {
  constructor(sky, directory, run) {
    this.sky = sky;
    this.directory = directory;
    this.run = run;
    this.window = null;
    this.observation = null;
    this.evidence = null;
  }
  static async open(sky, directory) {
    const run = JSON.parse((await fs.readFile(path.join(directory, 'run.json'), 'utf8')).replace(/^\uFEFF/, ''));
    if (run.protocol !== 'egaroucid-desktop-check-v1') throw new Error('Incompatible desktop run');
    return new DesktopDriver(sky, directory, run);
  }
  async select(returnedWindow) {
    const windows = await this.sky.list_windows();
    const matches = windows.filter(w => w.id === returnedWindow.id && w.app === returnedWindow.app);
    if (matches.length !== 1) throw new Error('Target window is no longer unique/current');
    this.window = await this.sky.get_window(matches[0]);
    return this.activate();
  }
  async activate() {
    if (!this.window) throw new Error('Select a unique returned window first');
    this.observation = null;
    await this.sky.activate_window({window: this.window});
    if (this.evidence) {
      this.evidence.actions.push({driver: 'os-input', type: 'activate', target: this.window,
        completed: true, time: new Date().toISOString()});
      await this.save();
    }
    return this.observe();
  }
  async observe(includeText = false) {
    this.observation = null;
    this.observation = await this.sky.get_window_state({window: this.window, include_screenshot: true, include_text: includeText});
    this.window = this.observation.window;
    return this.observation;
  }
  begin(name) {
    const spec = this.run.cases[name];
    if (!spec) throw new Error('Unknown desktop case: ' + name);
    this.evidence = {token: this.run.token, case: name, status: 'running', started_at: new Date().toISOString(), actions: [], assertions: []};
  }
  async act(action) {
    if (!this.evidence || !this.observation) throw new Error('Begin a case and inspect a fresh observation first');
    const observation = this.observation;
    const window = observation.window;
    const screenshotId = observation.screenshots?.[0]?.id;
    this.observation = null; // An uncertain input outcome must be reobserved.
    const {type, ...parameters} = action;
    const event = {driver: 'os-input', type, parameters, target: window, screenshotId, time: new Date().toISOString()};
    this.evidence.actions.push(event);
    try {
      if (type === 'click') await this.sky.click({window, screenshotId, ...parameters});
      else if (type === 'key') await this.sky.press_key({window, ...parameters});
      else if (type === 'drag') await this.sky.drag({window, screenshotId, ...parameters});
      else if (type === 'scroll') await this.sky.scroll({window, screenshotId, ...parameters});
      else if (type === 'text') {
        if (!observation.accessibility?.focused_element) throw new Error('Observe text-entry focus before typing');
        await this.sky.type_text({window, ...parameters});
      } else throw new Error('Unsupported OS input action: ' + type);
      event.completed = true;
      await this.save();
      // Allow the ordinary scene fade and one render before taking evidence.
      await new Promise(resolve => setTimeout(resolve, 200));
      return await this.observe();
    } catch (error) {
      event.error = String(error);
      this.evidence.status = 'failed';
      await this.save();
      throw error;
    }
  }
  async state() {
    let lastError;
    for (let attempt = 0; attempt < 20; attempt++) {
      try {
        const state = JSON.parse((await fs.readFile(path.join(this.directory, 'desktop-state.json'), 'utf8')).replace(/^\uFEFF/, ''));
        if (state.token !== this.run.token || state.marker !== this.run.protocol) throw new Error('Stale desktop state');
        return state;
      } catch (error) {
        lastError = error;
        await new Promise(resolve => setTimeout(resolve, 50));
      }
    }
    throw lastError;
  }
  async close() {
    if (!this.evidence || !this.observation) throw new Error('Observe the target before closing');
    const target = this.observation.window;
    this.observation = null;
    const event = {driver: 'os-input', type: 'close', target, time: new Date().toISOString()};
    this.evidence.actions.push(event);
    try {
      await this.sky.press_key({window: target, key: 'Alt_L+F4'});
    } catch (error) {
      // The adapter can lose the process while the requested window closes.
      // Verify disappearance instead of treating that adapter error as success.
      event.adapter_message = String(error);
    }
    event.completed = false;
    for (let attempt = 0; attempt < 20; attempt++) {
      const windows = await this.sky.list_windows();
      if (!windows.some(w => w.id === target.id && w.app === target.app)) {
        event.completed = true;
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    if (!event.completed) this.evidence.status = 'failed';
    await this.save();
    if (!event.completed) throw new Error('GUI window did not close');
  }
  async expect(name, expected) {
    let state;
    let passed = false;
    for (let attempt = 0; attempt < 30; attempt++) {
      state = await this.state();
      passed = Object.entries(expected).every(([key, value]) => JSON.stringify(state[key]) === JSON.stringify(value));
      if (passed) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    this.evidence.assertions.push({name, passed, expected, observed: state, afterAction: this.evidence.actions.length});
    if (!passed) this.evidence.status = 'failed';
    await this.save();
    if (!passed) throw new Error('Desktop postcondition failed: ' + name);
    return state;
  }
  async expectJsonFile(name, relativePath, expected) {
    const file = path.resolve(this.directory, relativePath);
    if (!file.startsWith(path.resolve(this.directory) + path.sep)) throw new Error('Evidence file must be inside the test directory');
    const buffer = await fs.readFile(file);
    const value = JSON.parse(buffer.toString('utf8').replace(/^\uFEFF/, ''));
    const stat = await fs.stat(file);
    const started = Date.parse(this.evidence.started_at || this.evidence.actions[0]?.time);
    const passed = stat.mtimeMs >= started && Object.entries(expected).every(([key, wanted]) =>
      JSON.stringify(value[key]) === JSON.stringify(wanted));
    this.evidence.assertions.push({name, passed, kind: 'json-file', expected: {file: relativePath, fields: expected},
      observed: {value, sha256: createHash('sha256').update(buffer).digest('hex'), modified: stat.mtimeMs},
      afterAction: this.evidence.actions.length});
    if (!passed) this.evidence.status = 'failed';
    await this.save();
    if (!passed) throw new Error('Saved file postcondition failed: ' + name);
    return value;
  }
  async expectFile(name, relativePath, expected) {
    const file = path.resolve(this.directory, relativePath);
    if (!file.startsWith(path.resolve(this.directory) + path.sep)) throw new Error('Evidence file must be inside the test directory');
    const buffer = await fs.readFile(file);
    const observed = {bytes: buffer.length, sha256: createHash('sha256').update(buffer).digest('hex'),
      text: buffer.toString('utf8').replace(/^\uFEFF/, '')};
    const passed = Object.entries(expected).every(([key, value]) => observed[key] === value);
    this.evidence.assertions.push({name, passed, kind: 'file', expected: {file: relativePath, fields: expected},
      observed, afterAction: this.evidence.actions.length});
    if (!passed) this.evidence.status = 'failed';
    await this.save();
    if (!passed) throw new Error('File postcondition failed: ' + name);
    return observed;
  }
  async save() {
    if (!this.evidence) return;
    await fs.writeFile(path.join(this.directory, this.evidence.case + '.result.json'), JSON.stringify(this.evidence, null, 2));
  }
  async complete() {
    const required = this.run.cases[this.evidence.case].required;
    const completed = new Set(this.evidence.assertions.filter(item => item.passed).map(item => item.name));
    if (this.evidence.status === 'failed' || !this.evidence.actions.length || required.some(item => !completed.has(item))) {
      throw new Error('Required OS actions/postconditions remain unverified');
    }
    this.evidence.status = 'passed';
    await this.save();
    return this.evidence;
  }
}
