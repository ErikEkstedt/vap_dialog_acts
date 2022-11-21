import torch
from torch.utils.data import Dataset
from os.path import expanduser, join
import matplotlib.pyplot as plt

from vap.utils import read_json, batch_to_device, everything_deterministic
from vap.model import VAPModel

from vap.audio import load_waveform, get_audio_info

from vap_agent.visualize_turn import plot_waveform, plot_mel_spectrogram


everything_deterministic()

torch.manual_seed(0)


REL_PATH = read_json("data/relative_audio_path.json")
AUDIO_ROOT = join(expanduser("~"), "projects/data/switchboard/audio")
CONTEXT = 20  # seconds of context prior to end
POST = 5  # seconds after the end of the utterance


def load_da_audio(da, context=CONTEXT, post=POST):
    audio_path = join(AUDIO_ROOT, REL_PATH[da["session"]] + ".wav")
    duration = get_audio_info(audio_path)["duration"]

    start_time = da["end"] - context
    if start_time < 0:
        start_time = 0

    end_time = da["end"] + post
    if end_time > duration:
        end_time = duration

    waveform, _ = load_waveform(
        audio_path,
        sample_rate=model.sample_rate,
        start_time=start_time,
        end_time=end_time,
        mono=False,
    )
    return waveform.unsqueeze(0)


def get_zones(p, x_frames):
    # Speaker B Zone
    blue = p[:, 0] >= 0.6
    x_blue = x_frames[blue]
    y_blue = p[blue][..., 0].cpu()

    # Speaker B Zone
    orange = p[:, 0] <= 0.4
    x_orange = x_frames[orange]
    y_orange = p[orange][..., 0].cpu()

    # NETRURAL ZONE
    neutral = torch.logical_and(p[:, 0] > 0.4, p[:, 0] < 0.6)
    x_neutral = x_frames[neutral]
    y_neutral = p[neutral][..., 0].cpu()

    return {
        "A": {"x": x_blue, "y": y_blue},
        "B": {"x": x_orange, "y": y_orange},
        "N": {"x": x_neutral, "y": y_neutral},
    }


def plot_model_probs_out(out, b=0, context=20, plot=True, s=10):
    n_frames = out["vad"].shape[1]
    x_frames = torch.arange(n_frames) / model.frame_hz

    # idx = torch.where(x_frames == context)[0]
    fig, axs = plt.subplots(6, 1, figsize=(16, 10), sharex=True)
    plot_mel_spectrogram(waveform[b], ax=[axs[0], axs[1]])
    axs[0].plot(x_frames, out["vad"][b, :, 0].cpu() * 80, color="w", label="VAD")
    axs[1].plot(x_frames, out["vad"][b, :, 1].cpu() * 80, color="w", label="VAD")
    plot_waveform(waveform[b], max_points=2000, ax=axs[2])

    # Plot probs

    now_zones = get_zones(out["p_now"][b], x_frames)
    axs[3].scatter(
        now_zones["A"]["x"], now_zones["A"]["y"], label="A now", color="b", s=s
    )
    axs[3].scatter(
        now_zones["B"]["x"], now_zones["B"]["y"], label="B now", color="orange", s=s
    )
    axs[3].scatter(
        now_zones["N"]["x"], now_zones["N"]["y"], label="N now", color="g", s=s
    )
    axs[3].axhline(0.5, color="b", linestyle="dotted")
    axs[3].axhline(0.4, color="c", linestyle="dotted")
    axs[3].axhline(0.6, color="c", linestyle="dotted")

    fut_zones = get_zones(out["p_future"][b], x_frames)
    axs[4].scatter(
        fut_zones["A"]["x"], fut_zones["A"]["y"], label="A future", color="b", s=s
    )
    axs[4].scatter(
        fut_zones["B"]["x"], fut_zones["B"]["y"], label="B future", color="orange", s=s
    )
    axs[4].scatter(
        fut_zones["N"]["x"], fut_zones["N"]["y"], label="N future", color="g", s=s
    )
    axs[4].axhline(0.5, color="b", linestyle="dotted")
    axs[4].axhline(0.4, color="c", linestyle="dotted")
    axs[4].axhline(0.6, color="c", linestyle="dotted")

    # axs[3].plot(x_frames, out["p_now"][b, :, 0].cpu(), color="r", linewidth=2)
    # axs[3].plot(x_frames, out["p_future"][b, :, 0].cpu(), color="r", linestyle="dashed", linewidth=2)
    # axs[4].plot(x_frames, out["p_now"][b, :, 1].cpu(), color="orange")
    # axs[4].plot(
    #     x_frames, out["p_future"][b, :, 1].cpu(), color="orange", linestyle="dashed"
    # )

    for a in axs:
        a.axvline(context, color="r", linewidth=2)

    # Plot Entropy
    axs[-1].plot(x_frames, out["H"][b].cpu(), color="g", label="Entropy, H")
    axs[-1].set_ylim([0, 8])
    for a in axs[:-1]:
        a.set_yticks([])

    for a in axs:  # not mels
        a.legend(loc="upper left", fontsize=12)
    plt.subplots_adjust(
        left=0.03, bottom=None, right=0.99, top=0.99, wspace=0.01, hspace=0
    )
    if plot:
        plt.pause(0.1)
    return fig, axs


# TODO:
class DADataset(Dataset):
    def __init__(self, filepath="data/da_utterances.json"):
        self.data = read_json(filepath)
        self.n_total = self._get_n_utterances()

    def _get_n_utterances(self):
        n = 0
        for da, utterances in self.data.items():
            n += len(utterances)
        return n

    def __len__(self):
        return self.n_total

    def __getitem__(self, idx):
        pass


if __name__ == "__main__":

    filepath = "data/da_utterances.json"
    data = read_json(filepath)

    print("Load Model...")
    model = VAPModel.load_from_checkpoint(
        "../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt"
    )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("CUDA")

    da = data["qy"][2]

    for da in data["qw"]:
        plt.close("all")
        waveform = load_da_audio(da)
        out = model.probs(
            waveform.to(model.device), now_lims=[0, 1], future_lims=[2, 3]
        )
        fig, ax = plot_model_probs_out(out)
        input()
