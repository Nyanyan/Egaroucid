// Japanese UI, window scale 1.0. These steps are executed one at a time:
// inspect the refreshed screenshot after every nextScenarioStep() call.
const click = (x, y, checkpoint) => ({action: {type: 'click', x, y}, checkpoint});
const key = (key, checkpoint) => ({action: {type: 'key', key}, checkpoint});
const main = {scene: 'Main_scene', loading: false};
const exportBook = () => [click(407, 39), click(423, 95), click(534, 115),
  click(612, 475, {name: 'returned-after-export', expected: main})];
const operation = (row, scene, startX = 401, startY = 403) => [
  click(407, 39), click(423, 76),
  click(535, row, {name: 'operation-screen', expected: {scene}}),
  click(startX, startY, {name: startX === 401 ? 'completed' : 'started',
    expected: startX === 401 ? {...main, book_changed: true} : {scene}}),
  ...(startX === 401 ? [] : [{...click(621, 475, {name: 'completed', expected: {...main, book_changed: true}}),
    before: {scene, operation_status: 'done'}}]),
  ...exportBook(),
];

export const scenarios = {
  'mouse-gameplay': [
    key('Control_L+n', {name: 'initial', expected: {...main, discs: 4, history_size: 1}}),
    click(317, 314, {name: 'legal-move', expected: {discs: 5, history_size: 2}}),
    click(65, 114, {name: 'illegal-move', expected: {discs: 5, history_size: 2}}),
    click(260, 39), click(275, 95, {name: 'back', expected: {discs: 4, history_size: 2}}),
    click(276, 76, {name: 'forward', expected: {discs: 5, history_size: 2}}), click(451, 502),
  ],
  'book-fix': operation(153, 'Fix_book'),
  'book-fix-edax': operation(171, 'Fix_book_edax'),
  'book-reduce': operation(191, 'Reduce_book', 621, 409),
  'book-leaves': operation(210, 'Leaf_recalculate_book', 621, 409),
  'book-lines': operation(229, 'N_lines_recalculate_book'),
  'book-store': operation(134, 'Store_book', 621, 409),
};

export function beginScenario(driver, name) {
  if (!scenarios[name]) throw new Error('Unknown desktop scenario: ' + name);
  driver.begin(name);
  driver.evidence.scenario = {name, next: 0};
}

export async function nextScenarioStep(driver) {
  const progress = driver.evidence?.scenario;
  if (!progress) throw new Error('Begin a scenario first');
  const step = scenarios[progress.name][progress.next];
  if (!step) return {finished: true, remaining: 0};
  const state = await driver.state();
  if (state.loading || state.ime_composition) throw new Error('Wait for loading or finish IME composition first');
  const screenshot = driver.observation?.screenshots?.[0];
  if (screenshot?.width !== 803 || ![530, 531].includes(screenshot?.height)) {
    throw new Error('Scenario requires the observed 803x530/531 Japanese GUI at scale 1.0');
  }
  if (step.before) await driver.expect('task-finished-before-return', step.before);
  await driver.act(step.action);
  progress.next++;
  await driver.save();
  if (step.checkpoint) await driver.expect(step.checkpoint.name, step.checkpoint.expected);
  return {finished: progress.next === scenarios[progress.name].length,
    remaining: scenarios[progress.name].length - progress.next};
}
