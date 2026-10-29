import test from 'node:test';
import assert from 'node:assert/strict';
import {beginScenario, nextScenarioStep} from './desktop_scenarios.mjs';

function fakeDriver() {
  return {
    observation: {screenshots: [{width: 803, height: 531}]},
    begin(name) { this.evidence = {case: name}; },
    async state() { return {loading: false, ime_composition: ''}; },
    async act() { this.calls++; },
    async expect() {}, async save() {}, calls: 0,
  };
}

test('one step issues exactly one action', async () => {
  const driver = fakeDriver();
  beginScenario(driver, 'mouse-gameplay');
  await nextScenarioStep(driver);
  assert.equal(driver.calls, 1);
  assert.equal(driver.evidence.scenario.next, 1);
});

test('unobserved or resized windows receive no input', async () => {
  for (const observation of [null, {screenshots: [{width: 1200, height: 900}]}]) {
    const driver = fakeDriver();
    driver.observation = observation;
    beginScenario(driver, 'mouse-gameplay');
    await assert.rejects(nextScenarioStep(driver), /803x530/);
    assert.equal(driver.calls, 0);
  }
});

test('a failed action does not advance the scenario', async () => {
  const driver = fakeDriver();
  driver.act = async () => { throw new Error('input failed'); };
  beginScenario(driver, 'mouse-gameplay');
  await assert.rejects(nextScenarioStep(driver), /input failed/);
  assert.equal(driver.evidence.scenario.next, 0);
});

test('loading and active IME composition block scenario input', async () => {
  for (const state of [{loading: true}, {ime_composition: 'にほんご'}]) {
    const driver = fakeDriver();
    driver.state = async () => state;
    beginScenario(driver, 'mouse-gameplay');
    await assert.rejects(nextScenarioStep(driver), /IME/);
    assert.equal(driver.calls, 0);
  }
});
