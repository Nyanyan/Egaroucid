// Actual desktop input driver. Inject the supported @oai/sky client from the
// computer-use runtime. Each act() performs ONE OS action and then refreshes;
// the caller inspects that observation before deciding the next action.
import fs from 'node:fs/promises';
import path from 'node:path';

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
    this.evidence = {token: this.run.token, case: name, status: 'running', actions: [], assertions: []};
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
  async expect(name, expected) {
    let state;
    let passed = false;
    for (let attempt = 0; attempt < 30; attempt++) {
      state = await this.state();
      passed = Object.entries(expected).every(([key, value]) => JSON.stringify(state[key]) === JSON.stringify(value));
      if (passed) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    this.evidence.assertions.push({name, passed, expected, observed: state});
    if (!passed) this.evidence.status = 'failed';
    await this.save();
    if (!passed) throw new Error('Desktop postcondition failed: ' + name);
    return state;
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
